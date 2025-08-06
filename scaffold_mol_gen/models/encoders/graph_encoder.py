"""
图编码器模块
使用GIN (Graph Isomorphism Network) 处理分子图结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.data import Data, Batch
import logging
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

class GINEncoder(nn.Module):
    """
    GIN (Graph Isomorphism Network) 编码器
    用于编码分子的图结构表示
    """
    
    def __init__(self,
                 input_dim: int = 9,  # 原子特征维度
                 hidden_dim: int = 768,
                 output_dim: int = 768,
                 num_layers: int = 5,
                 dropout: float = 0.1,
                 pool_type: str = 'mean',
                 virtual_node: bool = False):
        """
        初始化GIN编码器
        
        Args:
            input_dim: 输入节点特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            num_layers: GIN层数
            dropout: dropout比率
            pool_type: 池化类型 ('mean', 'add', 'max')
            virtual_node: 是否使用虚拟节点
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.pool_type = pool_type
        self.virtual_node = virtual_node
        
        # 初始节点嵌入
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        
        # GIN层
        self.gin_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            self.gin_layers.append(GINConv(mlp))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # 虚拟节点（可选）
        if virtual_node:
            self.virtual_node_embedding = nn.Parameter(torch.randn(1, hidden_dim))
            self.virtual_node_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        # 输出投影
        if hidden_dim != output_dim:
            self.output_projection = nn.Linear(hidden_dim, output_dim)
        else:
            self.output_projection = nn.Identity()
        
        # 额外的图级别特征处理
        self.graph_mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, edge_index, batch=None, edge_attr=None):
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
            batch: 批次索引 [num_nodes]
            edge_attr: 边特征（可选）
            
        Returns:
            node_features: 节点级别特征 [num_nodes, output_dim]
            graph_features: 图级别特征 [batch_size, output_dim]
        """
        # 初始节点嵌入
        h = self.node_embedding(x)
        
        # 虚拟节点初始化
        if self.virtual_node and batch is not None:
            batch_size = batch.max().item() + 1
            virtual_node_feat = self.virtual_node_embedding.repeat(batch_size, 1)
        
        # GIN层
        for i, (gin_layer, batch_norm) in enumerate(zip(self.gin_layers, self.batch_norms)):
            h_prev = h
            
            # GIN卷积
            h = gin_layer(h, edge_index)
            h = batch_norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            
            # 残差连接
            if i > 0:
                h = h + h_prev
            
            # 虚拟节点更新
            if self.virtual_node and batch is not None:
                # 聚合节点特征到虚拟节点
                temp = global_mean_pool(h, batch)
                virtual_node_feat = virtual_node_feat + self.virtual_node_mlp(temp)
                
                # 广播虚拟节点特征回节点
                h = h + virtual_node_feat[batch]
        
        # 输出投影
        node_features = self.output_projection(h)
        
        # 图级别池化
        if batch is not None:
            if self.pool_type == 'mean':
                graph_features = global_mean_pool(node_features, batch)
            elif self.pool_type == 'add':
                graph_features = global_add_pool(node_features, batch)
            elif self.pool_type == 'max':
                graph_features = global_max_pool(node_features, batch)
            else:
                raise ValueError(f"Unknown pool type: {self.pool_type}")
            
            # 图级别MLP
            graph_features = self.graph_mlp(graph_features)
        else:
            # 单图情况
            if self.pool_type == 'mean':
                graph_features = node_features.mean(dim=0, keepdim=True)
            elif self.pool_type == 'add':
                graph_features = node_features.sum(dim=0, keepdim=True)
            elif self.pool_type == 'max':
                graph_features = node_features.max(dim=0, keepdim=True)[0]
            
            graph_features = self.graph_mlp(graph_features)
        
        return node_features, graph_features
    
    def encode_graphs(self, graph_list):
        """
        编码图列表 - 兼容PyTorch Geometric 2.6.1
        
        Args:
            graph_list: PyG Data对象列表
            
        Returns:
            graph_features: 图级别特征 [batch_size, output_dim]
        """
        import torch
        from torch_geometric.nn import global_mean_pool
        
        # 如果列表为空，返回空张量
        if not graph_list:
            output_dim = self.output_projection.out_features if hasattr(self.output_projection, 'out_features') else 768
            return torch.empty(0, output_dim, device=next(self.parameters()).device)
        
        # 手动批处理，避免使用Batch.from_data_list
        device = next(self.parameters()).device
        
        # 收集所有图的数据
        all_x = []
        all_edge_index = []
        all_edge_attr = []
        all_batch = []
        
        node_offset = 0
        for i, data in enumerate(graph_list):
            # 节点特征
            x = data.x
            all_x.append(x)
            
            # 边索引（需要添加偏移量）
            edge_index = data.edge_index
            if edge_index.numel() > 0:  # 如果有边
                edge_index = edge_index + node_offset
            all_edge_index.append(edge_index)
            
            # 边特征（如果存在）
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                all_edge_attr.append(data.edge_attr)
            
            # 批次索引（标记每个节点属于哪个图）
            batch_idx = torch.full((x.size(0),), i, dtype=torch.long)
            all_batch.append(batch_idx)
            
            # 更新节点偏移量
            node_offset += x.size(0)
        
        # 拼接所有数据
        x = torch.cat(all_x, dim=0).to(device)
        edge_index = torch.cat(all_edge_index, dim=1).to(device) if all_edge_index else torch.empty((2, 0), dtype=torch.long, device=device)
        batch = torch.cat(all_batch, dim=0).to(device)
        
        # 处理边特征 - 处理维度不一致的情况
        edge_attr = None
        if all_edge_attr:
            # 检查是否所有边特征都有相同维度
            has_features = [e for e in all_edge_attr if e.numel() > 0]
            if has_features:
                # 如果有些有特征有些没有，只使用有特征的
                edge_attr = torch.cat(has_features, dim=0).to(device) if has_features else None
            else:
                edge_attr = None
        
        # 通过GIN网络编码
        with torch.no_grad():
            # 调用forward方法
            _, graph_features = self.forward(x, edge_index, batch, edge_attr)
        
        return graph_features

class GraphFeatureExtractor:
    """
    图特征提取器
    将SMILES转换为图特征
    """
    
    @staticmethod
    def smiles_to_graph(smiles: str) -> Data:
        """
        将SMILES转换为PyG图数据
        
        Args:
            smiles: SMILES字符串
            
        Returns:
            graph: PyG Data对象
        """
        from rdkit import Chem
        import numpy as np
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # 返回空图
            return Data(
                x=torch.zeros((1, 9)),
                edge_index=torch.zeros((2, 0), dtype=torch.long)
            )
        
        # 提取节点特征
        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                atom.GetTotalNumHs(),
                atom.GetNumRadicalElectrons(),
                int(atom.IsInRing()),
                int(atom.GetChiralTag())
            ]
            atom_features.append(features)
        
        # 提取边
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.extend([[i, j], [j, i]])  # 无向图
        
        # 转换为tensor
        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        
        return Data(x=x, edge_index=edge_index)
    
    @staticmethod
    def batch_smiles_to_graphs(smiles_list: List[str]) -> List[Data]:
        """
        批量转换SMILES为图
        
        Args:
            smiles_list: SMILES字符串列表
            
        Returns:
            graphs: PyG Data对象列表
        """
        extractor = GraphFeatureExtractor()
        graphs = []
        for smiles in smiles_list:
            graph = extractor.smiles_to_graph(smiles)
            graphs.append(graph)
        return graphs