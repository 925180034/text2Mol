"""
简化的模态融合层
专注于Scaffold-Text融合，为MolT5生成提供统一特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class SimplifiedCrossModalAttention(nn.Module):
    """
    简化版跨模态注意力机制
    使用Scaffold特征作为Query，Text特征作为Key/Value
    """
    
    def __init__(self, hidden_size: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # 多头注意力
        self.attention = nn.MultiheadAttention(
            hidden_size, 
            num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, scaffold_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scaffold_feat: [batch_size, hidden_size]
            text_feat: [batch_size, hidden_size]
            
        Returns:
            enhanced_features: [batch_size, hidden_size]
        """
        # 扩展维度以适应注意力机制
        scaffold_seq = scaffold_feat.unsqueeze(1)  # [batch, 1, hidden]
        text_seq = text_feat.unsqueeze(1)  # [batch, 1, hidden]
        
        # 跨模态注意力
        attn_out, attn_weights = self.attention(
            query=scaffold_seq,
            key=text_seq,
            value=text_seq
        )
        
        # 残差连接和归一化
        scaffold_enhanced = self.norm1(scaffold_seq + attn_out)
        
        # 前馈网络
        ffn_out = self.ffn(scaffold_enhanced)
        output = self.norm2(scaffold_enhanced + ffn_out)
        
        # 压缩序列维度
        return output.squeeze(1)


class GatedFusion(nn.Module):
    """
    门控融合机制
    动态学习Scaffold和Text特征的重要性权重
    """
    
    def __init__(self, hidden_size: int = 768, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 门控网络
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        # 融合投影
        self.fusion_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
    def forward(self, scaffold_feat: torch.Tensor, text_feat: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            scaffold_feat: [batch_size, hidden_size]
            text_feat: [batch_size, hidden_size]
            
        Returns:
            fused_features: [batch_size, hidden_size]
            gate_info: 门控权重信息
        """
        # 计算门控权重
        concat_features = torch.cat([scaffold_feat, text_feat], dim=-1)
        gate_weights = self.gate_network(concat_features)
        
        # 应用门控融合
        fused = gate_weights * scaffold_feat + (1 - gate_weights) * text_feat
        
        # 最终投影
        output = self.fusion_projection(fused)
        
        # 返回融合特征和门控信息
        gate_info = {
            'scaffold_weight': gate_weights.mean().item(),
            'text_weight': (1 - gate_weights).mean().item()
        }
        
        return output, gate_info


class ModalFusionLayer(nn.Module):
    """
    完整的模态融合层
    整合跨模态注意力和门控融合
    """
    
    def __init__(self, 
                 hidden_size: int = 768,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 fusion_type: str = 'both'):
        """
        Args:
            hidden_size: 隐藏层维度
            num_heads: 注意力头数
            dropout: Dropout率
            fusion_type: 融合类型 ('attention', 'gated', 'both')
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.fusion_type = fusion_type
        
        # 特征投影层（确保维度对齐）
        self.scaffold_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        self.text_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        # 融合组件
        if fusion_type in ['attention', 'both']:
            self.cross_attention = SimplifiedCrossModalAttention(
                hidden_size, num_heads, dropout
            )
        
        if fusion_type in ['gated', 'both']:
            self.gated_fusion = GatedFusion(hidden_size, dropout)
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        logger.info(f"初始化模态融合层: type={fusion_type}, hidden_size={hidden_size}")
        
    def forward(self, 
                scaffold_features: torch.Tensor,
                text_features: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        前向传播
        
        Args:
            scaffold_features: [batch_size, hidden_size] Scaffold编码特征
            text_features: [batch_size, hidden_size] 文本编码特征
            
        Returns:
            fused_features: [batch_size, hidden_size] 融合后的特征
            fusion_info: 融合过程信息
        """
        # 1. 特征投影
        scaffold = self.scaffold_projection(scaffold_features)
        text = self.text_projection(text_features)
        
        fusion_info = {}
        
        # 2. 应用融合策略
        if self.fusion_type == 'attention':
            # 仅使用注意力机制
            fused = self.cross_attention(scaffold, text)
            
        elif self.fusion_type == 'gated':
            # 仅使用门控融合
            fused, gate_info = self.gated_fusion(scaffold, text)
            fusion_info.update(gate_info)
            
        elif self.fusion_type == 'both':
            # 先注意力增强，再门控融合
            scaffold_enhanced = self.cross_attention(scaffold, text)
            fused, gate_info = self.gated_fusion(scaffold_enhanced, text)
            fusion_info.update(gate_info)
            
        else:
            # 简单平均（fallback）
            fused = (scaffold + text) / 2
        
        # 3. 输出投影
        output = self.output_projection(fused)
        
        # 4. 添加统计信息
        fusion_info.update({
            'output_norm': output.norm(dim=-1).mean().item(),
            'scaffold_norm': scaffold.norm(dim=-1).mean().item(),
            'text_norm': text.norm(dim=-1).mean().item()
        })
        
        return output, fusion_info


class MultiModalFusionLayer(nn.Module):
    """
    多模态融合层（支持多种Scaffold模态）
    可以处理不同的Scaffold输入模态
    """
    
    def __init__(self,
                 hidden_size: int = 768,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 为每种Scaffold模态创建独立的融合层
        self.fusion_layers = nn.ModuleDict({
            'smiles': ModalFusionLayer(hidden_size, num_heads, dropout, 'both'),
            'graph': ModalFusionLayer(hidden_size, num_heads, dropout, 'both'),
            'image': ModalFusionLayer(hidden_size, num_heads, dropout, 'both')
        })
        
        # 模态选择网络（可选，用于自适应选择）
        self.modality_selector = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),  # 3种模态
            nn.Softmax(dim=-1)
        )
        
    def forward(self,
                scaffold_features: torch.Tensor,
                text_features: torch.Tensor,
                scaffold_modality: str = 'smiles') -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            scaffold_features: Scaffold特征
            text_features: 文本特征
            scaffold_modality: Scaffold的模态类型
            
        Returns:
            fused_features: 融合后的特征
            fusion_info: 融合信息
        """
        if scaffold_modality not in self.fusion_layers:
            raise ValueError(f"不支持的Scaffold模态: {scaffold_modality}")
        
        # 使用对应的融合层
        fusion_layer = self.fusion_layers[scaffold_modality]
        fused_features, fusion_info = fusion_layer(scaffold_features, text_features)
        
        # 添加模态信息
        fusion_info['scaffold_modality'] = scaffold_modality
        
        return fused_features, fusion_info


def test_fusion_layer():
    """测试融合层功能"""
    import torch
    
    # 设置随机种子
    torch.manual_seed(42)
    
    # 创建测试数据
    batch_size = 2
    hidden_size = 768
    
    scaffold_features = torch.randn(batch_size, hidden_size)
    text_features = torch.randn(batch_size, hidden_size)
    
    # 测试不同融合类型
    for fusion_type in ['attention', 'gated', 'both']:
        print(f"\n测试融合类型: {fusion_type}")
        fusion_layer = ModalFusionLayer(
            hidden_size=hidden_size,
            fusion_type=fusion_type
        )
        
        # 前向传播
        fused, info = fusion_layer(scaffold_features, text_features)
        
        print(f"输出形状: {fused.shape}")
        print(f"融合信息: {info}")
        
        assert fused.shape == (batch_size, hidden_size), f"输出形状错误: {fused.shape}"
    
    # 测试多模态融合
    print("\n测试多模态融合层")
    multi_fusion = MultiModalFusionLayer(hidden_size=hidden_size)
    
    for modality in ['smiles', 'graph', 'image']:
        print(f"\n模态: {modality}")
        fused, info = multi_fusion(scaffold_features, text_features, modality)
        print(f"输出形状: {fused.shape}")
        print(f"融合信息: {info}")
    
    print("\n✅ 所有测试通过！")


if __name__ == "__main__":
    test_fusion_layer()