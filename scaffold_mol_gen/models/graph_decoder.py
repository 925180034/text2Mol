"""
分子图解码器
将768维特征向量解码为分子图结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class MolecularGraphDecoder(nn.Module):
    """
    分子图解码器
    将统一的768维特征向量解码为分子图结构
    """
    
    def __init__(self,
                 input_dim: int = 768,
                 max_atoms: int = 100,
                 num_atom_types: int = 119,  # 1-118号元素 + UNK
                 num_bond_types: int = 4,    # 单键、双键、三键、芳香键
                 hidden_dim: int = 512,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        """
        Args:
            input_dim: 输入特征维度
            max_atoms: 最大原子数
            num_atom_types: 原子类型数量
            num_bond_types: 键类型数量  
            hidden_dim: 隐藏层维度
            num_layers: 解码层数
            dropout: Dropout率
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.max_atoms = max_atoms
        self.num_atom_types = num_atom_types
        self.num_bond_types = num_bond_types
        self.hidden_dim = hidden_dim
        
        # 输入投影层
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 原子数量预测器
        self.atom_count_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, max_atoms + 1),  # +1 for 0 atoms
        )
        
        # 原子类型预测器 (每个原子位置)
        self.atom_type_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_atoms * num_atom_types)
        )
        
        # 键存在性预测器 (上三角矩阵)
        self.bond_existence_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, (max_atoms * (max_atoms - 1)) // 2)
        )
        
        # 键类型预测器
        self.bond_type_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, (max_atoms * (max_atoms - 1)) // 2 * num_bond_types)
        )
        
        # 原子特征增强器
        self.atom_feature_enhancer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        logger.info(f"初始化分子图解码器: {input_dim}→{hidden_dim}, max_atoms={max_atoms}")
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            features: [batch_size, 768] 输入特征
            
        Returns:
            dict包含预测的原子数量、原子类型、键存在性、键类型
        """
        batch_size = features.size(0)
        
        # 投影到隐藏维度
        hidden = self.input_projection(features)  # [batch_size, hidden_dim]
        
        # 预测原子数量
        atom_count_logits = self.atom_count_predictor(hidden)  # [batch_size, max_atoms+1]
        
        # 增强特征用于原子和键预测
        enhanced_hidden = hidden
        for enhancer in self.atom_feature_enhancer:
            enhanced_hidden = enhancer(enhanced_hidden) + enhanced_hidden  # 残差连接
        
        # 预测原子类型
        atom_type_logits = self.atom_type_predictor(enhanced_hidden)
        atom_type_logits = atom_type_logits.view(batch_size, self.max_atoms, self.num_atom_types)
        
        # 预测键存在性
        bond_existence_logits = self.bond_existence_predictor(enhanced_hidden)
        
        # 预测键类型
        bond_type_logits = self.bond_type_predictor(enhanced_hidden)
        bond_pairs = (self.max_atoms * (self.max_atoms - 1)) // 2
        bond_type_logits = bond_type_logits.view(batch_size, bond_pairs, self.num_bond_types)
        
        return {
            'atom_count_logits': atom_count_logits,
            'atom_type_logits': atom_type_logits,
            'bond_existence_logits': bond_existence_logits,
            'bond_type_logits': bond_type_logits
        }
    
    def decode_to_graphs(self, 
                        predictions: Dict[str, torch.Tensor],
                        threshold: float = 0.5,
                        temperature: float = 1.0) -> List[Data]:
        """
        将预测结果解码为分子图
        
        Args:
            predictions: 模型预测结果
            threshold: 键存在性阈值
            temperature: 采样温度
            
        Returns:
            PyTorch Geometric Data对象列表
        """
        batch_size = predictions['atom_count_logits'].size(0)
        graphs = []
        
        for i in range(batch_size):
            try:
                graph = self._decode_single_graph(
                    {k: v[i] for k, v in predictions.items()},
                    threshold=threshold,
                    temperature=temperature
                )
                graphs.append(graph)
            except Exception as e:
                logger.warning(f"解码第{i}个图失败: {e}")
                # 返回空图
                graphs.append(Data(x=torch.empty(0, self.num_atom_types), 
                                 edge_index=torch.empty(2, 0, dtype=torch.long)))
        
        return graphs
    
    def _decode_single_graph(self, 
                           pred: Dict[str, torch.Tensor],
                           threshold: float = 0.5,
                           temperature: float = 1.0) -> Data:
        """解码单个分子图"""
        
        # 1. 预测原子数量
        atom_count_probs = F.softmax(pred['atom_count_logits'] / temperature, dim=-1)
        num_atoms = torch.multinomial(atom_count_probs, 1).item()
        num_atoms = min(num_atoms, self.max_atoms)  # 限制最大原子数
        
        if num_atoms == 0:
            return Data(x=torch.empty(0, self.num_atom_types), 
                       edge_index=torch.empty(2, 0, dtype=torch.long))
        
        # 2. 预测原子类型
        atom_type_probs = F.softmax(pred['atom_type_logits'] / temperature, dim=-1)
        atom_types = torch.multinomial(atom_type_probs[:num_atoms], 1).squeeze(-1)
        
        # 创建原子特征 (one-hot编码)
        atom_features = F.one_hot(atom_types, num_classes=self.num_atom_types).float()
        
        # 3. 预测键
        edge_indices = []
        edge_types = []
        
        bond_existence_probs = torch.sigmoid(pred['bond_existence_logits'])
        bond_type_probs = F.softmax(pred['bond_type_logits'] / temperature, dim=-1)
        
        # 遍历所有可能的原子对
        bond_idx = 0
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                if bond_idx >= len(bond_existence_probs):
                    break
                
                # 判断是否存在键
                if bond_existence_probs[bond_idx] > threshold:
                    # 预测键类型
                    bond_type = torch.multinomial(bond_type_probs[bond_idx], 1).item()
                    
                    # 添加双向边
                    edge_indices.extend([[i, j], [j, i]])
                    edge_types.extend([bond_type, bond_type])
                
                bond_idx += 1
        
        # 构建边张量
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_types, dtype=torch.long)
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long)
            edge_attr = torch.empty(0, dtype=torch.long)
        
        return Data(x=atom_features, edge_index=edge_index, edge_attr=edge_attr)
    
    def generate_graphs(self,
                       features: torch.Tensor,
                       num_samples: int = 1,
                       threshold: float = 0.5,
                       temperature: float = 1.0) -> List[List[Data]]:
        """
        生成分子图
        
        Args:
            features: [batch_size, 768] 输入特征
            num_samples: 每个输入生成的样本数
            threshold: 键存在性阈值
            temperature: 采样温度
            
        Returns:
            批次中每个样本的生成图列表
        """
        self.eval()
        
        with torch.no_grad():
            all_samples = []
            
            for _ in range(num_samples):
                predictions = self(features)
                graphs = self.decode_to_graphs(
                    predictions, 
                    threshold=threshold, 
                    temperature=temperature
                )
                all_samples.append(graphs)
            
            # 重组为 [batch_size, num_samples]
            batch_samples = []
            for i in range(features.size(0)):
                batch_samples.append([samples[i] for samples in all_samples])
            
            return batch_samples


class GraphDecoderLoss(nn.Module):
    """分子图解码器损失函数"""
    
    def __init__(self,
                 atom_count_weight: float = 1.0,
                 atom_type_weight: float = 1.0,
                 bond_existence_weight: float = 1.0,
                 bond_type_weight: float = 1.0):
        super().__init__()
        
        self.atom_count_weight = atom_count_weight
        self.atom_type_weight = atom_type_weight
        self.bond_existence_weight = bond_existence_weight
        self.bond_type_weight = bond_type_weight
        
        self.atom_count_loss = nn.CrossEntropyLoss()
        self.atom_type_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.bond_existence_loss = nn.BCEWithLogitsLoss()
        self.bond_type_loss = nn.CrossEntropyLoss(ignore_index=-1)
    
    def forward(self, 
               predictions: Dict[str, torch.Tensor],
               targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算损失
        
        Args:
            predictions: 模型预测
            targets: 目标值
            
        Returns:
            总损失和各部分损失
        """
        losses = {}
        
        # 原子数量损失
        atom_count_loss = self.atom_count_loss(
            predictions['atom_count_logits'], 
            targets['atom_counts']
        )
        losses['atom_count'] = atom_count_loss.item()
        
        # 原子类型损失
        atom_type_loss = self.atom_type_loss(
            predictions['atom_type_logits'].view(-1, predictions['atom_type_logits'].size(-1)),
            targets['atom_types'].view(-1)
        )
        losses['atom_type'] = atom_type_loss.item()
        
        # 键存在性损失
        bond_existence_loss = self.bond_existence_loss(
            predictions['bond_existence_logits'],
            targets['bond_existence'].float()
        )
        losses['bond_existence'] = bond_existence_loss.item()
        
        # 键类型损失 (只在存在键的地方计算)
        bond_mask = targets['bond_existence'] == 1
        if bond_mask.any():
            masked_bond_type_logits = predictions['bond_type_logits'][bond_mask]
            masked_bond_type_targets = targets['bond_types'][bond_mask]
            
            bond_type_loss = self.bond_type_loss(
                masked_bond_type_logits,
                masked_bond_type_targets
            )
        else:
            bond_type_loss = torch.tensor(0.0, device=predictions['bond_type_logits'].device)
        
        losses['bond_type'] = bond_type_loss.item()
        
        # 总损失
        total_loss = (
            self.atom_count_weight * atom_count_loss +
            self.atom_type_weight * atom_type_loss +
            self.bond_existence_weight * bond_existence_loss +
            self.bond_type_weight * bond_type_loss
        )
        
        return total_loss, losses


def test_graph_decoder():
    """测试分子图解码器"""
    print("测试分子图解码器...")
    
    # 创建解码器
    decoder = MolecularGraphDecoder(
        input_dim=768,
        max_atoms=20,
        hidden_dim=256
    )
    
    # 测试输入
    batch_size = 3
    features = torch.randn(batch_size, 768)
    
    # 前向传播
    predictions = decoder(features)
    
    print("预测输出形状:")
    for key, value in predictions.items():
        print(f"  {key}: {value.shape}")
    
    # 测试解码
    graphs = decoder.decode_to_graphs(predictions)
    
    print(f"\n生成了 {len(graphs)} 个图:")
    for i, graph in enumerate(graphs):
        print(f"  图 {i}: {graph.x.shape[0]} 个原子, {graph.edge_index.shape[1]} 条边")
    
    # 测试生成
    generated_samples = decoder.generate_graphs(features, num_samples=2)
    print(f"\n生成了 {len(generated_samples)} 个批次的样本")
    
    print("✅ 分子图解码器测试完成")


if __name__ == "__main__":
    test_graph_decoder()