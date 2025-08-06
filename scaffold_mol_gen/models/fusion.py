"""
Advanced modal fusion layer for integrating multi-modal features.

This module implements sophisticated fusion mechanisms including:
- Cross-modal attention
- Gating mechanisms  
- Feature alignment
- Hierarchical fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math

class AdvancedModalFusion(nn.Module):
    """高级模态融合层，使用注意力机制和门控融合多模态特征"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.hidden_size = config.get('hidden_size', 768)
        self.num_heads = config.get('num_attention_heads', 12)
        self.num_fusion_layers = config.get('num_fusion_layers', 6)
        self.dropout = config.get('dropout', 0.1)
        
        # 模态位置嵌入
        self.modal_embeddings = nn.Embedding(4, self.hidden_size)  # 4种模态
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_size * 4,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_fusion_layers
        )
        
        # 门控机制 - 为每个模态学习重要性权重
        self.modal_gates = nn.ModuleDict({
            'scaffold': GatingNetwork(self.hidden_size),
            'text': GatingNetwork(self.hidden_size),
            'graph': GatingNetwork(self.hidden_size),
            'image': GatingNetwork(self.hidden_size)
        })
        
        # 跨模态注意力
        self.cross_modal_attention = nn.MultiheadAttention(
            self.hidden_size,
            self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # 分层融合网络
        self.hierarchical_fusion = HierarchicalFusion(self.hidden_size)
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )
        
        # 特征对齐模块
        self.feature_alignment = FeatureAlignment(self.hidden_size)
        
    def forward(self, features_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        融合多模态特征
        
        Args:
            features_dict: Dictionary containing features from different modalities
                - 'scaffold': [batch_size, hidden_size] - Scaffold features
                - 'text': [batch_size, hidden_size] - Text features  
                - 'graph': [batch_size, hidden_size] - Graph features (optional)
                - 'image': [batch_size, hidden_size] - Image features (optional)
                
        Returns:
            fused_features: [batch_size, hidden_size] - Fused representation
            modal_features: Dict with individual modal representations
        """
        batch_size = next(iter(features_dict.values())).shape[0]
        device = next(iter(features_dict.values())).device
        
        # 1. 收集并准备所有模态特征
        modal_features_list = []
        modal_types = []
        modal_positions = []
        
        modal_mapping = {'scaffold': 0, 'text': 1, 'graph': 2, 'image': 3}
        
        for modal_name, modal_idx in modal_mapping.items():
            if modal_name in features_dict and features_dict[modal_name] is not None:
                feat = features_dict[modal_name]
                
                # 添加模态位置嵌入
                modal_emb = self.modal_embeddings(
                    torch.tensor([modal_idx], device=device)
                ).expand(batch_size, -1)
                feat_with_pos = feat + modal_emb
                
                modal_features_list.append(feat_with_pos.unsqueeze(1))
                modal_types.append(modal_name)
                modal_positions.append(modal_idx)
        
        if not modal_features_list:
            raise ValueError("No input features provided")
        
        # 2. 拼接所有特征 [batch_size, num_modals, hidden_size]
        combined_features = torch.cat(modal_features_list, dim=1)
        
        # 3. 特征对齐
        aligned_features = self.feature_alignment(combined_features, modal_types)
        
        # 4. Transformer自注意力编码
        # 创建attention mask（所有位置都可见）
        seq_len = aligned_features.shape[1]
        src_mask = torch.zeros(seq_len, seq_len, device=device)
        
        encoded_features = self.transformer(
            aligned_features,
            mask=src_mask
        )
        
        # 5. 应用门控机制
        gated_features = []
        modal_weights = {}
        
        for i, modal_name in enumerate(modal_types):
            feat = encoded_features[:, i, :]
            gate_weight = self.modal_gates[modal_name](feat)
            gated_feat = feat * gate_weight
            gated_features.append(gated_feat)
            modal_weights[modal_name] = gate_weight.mean().item()
        
        # 6. 分层融合
        if len(gated_features) > 1:
            fused_features = self.hierarchical_fusion(gated_features, modal_types)
        else:
            fused_features = gated_features[0]
        
        # 7. 跨模态注意力增强
        if len(gated_features) > 1:
            # 使用主要模态（scaffold+text）作为query
            main_modals = [f for i, f in enumerate(gated_features) 
                          if modal_types[i] in ['scaffold', 'text']]
            if main_modals:
                query = torch.stack(main_modals, dim=1)  # [batch, num_main, hidden]
                key_value = torch.stack(gated_features, dim=1)  # [batch, num_all, hidden]
                
                enhanced_features, attention_weights = self.cross_modal_attention(
                    query, key_value, key_value
                )
                
                # 融合原始特征和增强特征
                fused_features = fused_features + enhanced_features.mean(dim=1)
        
        # 8. 输出投影
        final_features = self.output_projection(fused_features)
        
        # 创建输出字典
        output_dict = {
            modal_name: encoded_features[:, i, :] 
            for i, modal_name in enumerate(modal_types)
        }
        output_dict['modal_weights'] = modal_weights
        
        return final_features, output_dict


class GatingNetwork(nn.Module):
    """门控网络，为每个模态学习重要性权重"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, hidden_size]
        Returns:
            gate_weights: [batch_size, 1]
        """
        return self.gate(x)


class HierarchicalFusion(nn.Module):
    """分层融合网络，逐步融合不同模态"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 配对融合层
        self.pairwise_fusion = nn.ModuleDict()
        modality_pairs = [
            ('scaffold', 'text'),
            ('scaffold', 'graph'), 
            ('scaffold', 'image'),
            ('text', 'graph'),
            ('text', 'image'),
            ('graph', 'image')
        ]
        
        for pair in modality_pairs:
            self.pairwise_fusion[f"{pair[0]}_{pair[1]}"] = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.GELU(), 
                nn.LayerNorm(hidden_size)
            )
        
        # 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size)
        )
        
    def forward(self, features: List[torch.Tensor], modal_types: List[str]) -> torch.Tensor:
        """
        分层融合多个模态特征
        
        Args:
            features: List of feature tensors [batch_size, hidden_size]
            modal_types: List of modality names
            
        Returns:
            fused_features: [batch_size, hidden_size]
        """
        if len(features) == 1:
            return features[0]
        
        # 首先融合主要模态 (scaffold + text)
        scaffold_idx = modal_types.index('scaffold') if 'scaffold' in modal_types else None
        text_idx = modal_types.index('text') if 'text' in modal_types else None
        
        if scaffold_idx is not None and text_idx is not None:
            # 融合scaffold和text
            main_fusion = self.pairwise_fusion['scaffold_text'](
                torch.cat([features[scaffold_idx], features[text_idx]], dim=-1)
            )
            fused_result = main_fusion
        else:
            # 如果没有主要模态对，使用第一个特征作为基础
            fused_result = features[0]
        
        # 逐步融合其他模态
        for i, (feat, modal_type) in enumerate(zip(features, modal_types)):
            if modal_type not in ['scaffold', 'text'] or (scaffold_idx is None or text_idx is None):
                # 将当前模态与已融合的特征结合
                combined = torch.cat([fused_result, feat], dim=-1)
                fused_result = self.final_fusion(
                    nn.Linear(combined.shape[-1], self.hidden_size).to(combined.device)(combined)
                )
        
        return fused_result


class FeatureAlignment(nn.Module):
    """特征对齐模块，确保不同模态特征在相同语义空间"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 模态特定的对齐网络
        self.alignment_networks = nn.ModuleDict({
            'scaffold': nn.Linear(hidden_size, hidden_size),
            'text': nn.Linear(hidden_size, hidden_size),
            'graph': nn.Linear(hidden_size, hidden_size), 
            'image': nn.Linear(hidden_size, hidden_size)
        })
        
        # 对比学习头
        self.contrastive_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        
    def forward(self, features: torch.Tensor, modal_types: List[str]) -> torch.Tensor:
        """
        对齐不同模态的特征表示
        
        Args:
            features: [batch_size, num_modals, hidden_size]
            modal_types: List of modality type names
            
        Returns:
            aligned_features: [batch_size, num_modals, hidden_size]
        """
        aligned_features = []
        
        for i, modal_type in enumerate(modal_types):
            if modal_type in self.alignment_networks:
                aligned_feat = self.alignment_networks[modal_type](features[:, i, :])
            else:
                aligned_feat = features[:, i, :]
            aligned_features.append(aligned_feat.unsqueeze(1))
        
        return torch.cat(aligned_features, dim=1)
    
    def compute_alignment_loss(self, features: torch.Tensor, 
                              modal_types: List[str]) -> torch.Tensor:
        """计算模态对齐损失"""
        if len(modal_types) < 2:
            return torch.tensor(0.0, device=features.device)
        
        # 对比学习损失
        projected_features = []
        for i in range(features.shape[1]):
            projected = self.contrastive_head(features[:, i, :])
            projected_features.append(projected)
        
        # 计算模态间的相似性
        loss = 0.0
        num_pairs = 0
        
        for i in range(len(projected_features)):
            for j in range(i + 1, len(projected_features)):
                # 余弦相似性
                sim = F.cosine_similarity(
                    projected_features[i], 
                    projected_features[j], 
                    dim=-1
                ).mean()
                loss += (1 - sim)  # 最大化相似性
                num_pairs += 1
        
        return loss / num_pairs if num_pairs > 0 else torch.tensor(0.0)


class AttentionPooling(nn.Module):
    """注意力池化层"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            features: [batch_size, seq_len, hidden_size]
            mask: [batch_size, seq_len]
        """
        attention_weights = self.attention(features).squeeze(-1)  # [batch, seq_len]
        
        if mask is not None:
            attention_weights = attention_weights.masked_fill(~mask, -float('inf'))
        
        attention_weights = F.softmax(attention_weights, dim=-1).unsqueeze(-1)
        pooled = (features * attention_weights).sum(dim=1)
        
        return pooled