"""
Multi-modal encoders for scaffold-based molecular generation.

This module implements encoders for different input modalities:
- SMILES sequences (using BART)
- Text descriptions (using SciBERT)  
- Molecular graphs (using GIN)
- Molecular images (using Swin Transformer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, BertModel, AutoModel
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool
import timm
from typing import Optional, Dict, Any

class EnhancedMultiModalEncoders(nn.Module):
    """Enhanced multi-modal encoders with advanced architectures."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.hidden_size = config.get('hidden_size', 768)
        
        # SMILES编码器 (使用BART或专门的分子模型)
        self.smiles_encoder = AutoModel.from_pretrained(
            config.get('smiles_encoder_name', 'facebook/bart-base')
        )
        
        # 文本编码器 (SciBERT)
        self.text_encoder = BertModel.from_pretrained(
            config.get('text_encoder_name', 'allenai/scibert_scivocab_uncased')
        )
        
        # 图编码器 (GIN网络) 
        self.gin_conv_layers = nn.ModuleList()
        self.gin_bn_layers = nn.ModuleList()
        
        num_gin_layers = config.get('num_gin_layers', 3)
        for i in range(num_gin_layers):
            if i == 0:
                # 第一层：原子特征输入
                nn_layer = nn.Sequential(
                    nn.Linear(9, self.hidden_size),  # 9是原子特征维度
                    nn.BatchNorm1d(self.hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size, self.hidden_size)
                )
            else:
                nn_layer = nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.BatchNorm1d(self.hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size, self.hidden_size)
                )
            
            self.gin_conv_layers.append(GINConv(nn_layer))
            self.gin_bn_layers.append(nn.BatchNorm1d(self.hidden_size))
        
        # 图像编码器 (Swin Transformer V2)
        self.image_encoder = timm.create_model(
            config.get('image_encoder_name', 'swinv2_tiny_window8_256'),
            pretrained=True,
            num_classes=0  # 移除分类头
        )
        
        # 投影层，统一所有模态到相同维度
        self.projections = nn.ModuleDict({
            'smiles': nn.Linear(768, self.hidden_size),
            'text': nn.Linear(768, self.hidden_size), 
            'graph': nn.Linear(self.hidden_size * 2, self.hidden_size),  # concat pooling
            'image': nn.Linear(768, self.hidden_size)
        })
        
        # Dropout层
        self.dropout = nn.Dropout(config.get('dropout', 0.1))
        
    def encode_smiles(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        编码SMILES序列
        
        Args:
            input_ids: SMILES token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            encoded_features: [batch_size, hidden_size]
        """
        outputs = self.smiles_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 使用注意力加权平均池化
        hidden_states = outputs.last_hidden_state
        pooled = (hidden_states * attention_mask.unsqueeze(-1)).sum(1) / \
                 attention_mask.sum(-1, keepdim=True).clamp(min=1)
        
        return self.dropout(self.projections['smiles'](pooled))
    
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        编码文本描述
        
        Args:
            input_ids: Text token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            encoded_features: [batch_size, hidden_size]
        """
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 使用[CLS] token的表示
        pooled = outputs.pooler_output
        
        return self.dropout(self.projections['text'](pooled))
    
    def encode_graph(self, x: torch.Tensor, edge_index: torch.Tensor, 
                    batch: torch.Tensor) -> torch.Tensor:
        """
        编码分子图结构
        
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]
            
        Returns:
            encoded_features: [batch_size, hidden_size]
        """
        # 多层GIN卷积
        for i, (conv, bn) in enumerate(zip(self.gin_conv_layers, self.gin_bn_layers)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # 组合mean和max池化
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        
        return self.dropout(self.projections['graph'](pooled))
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        编码分子图像
        
        Args:
            images: Image tensor [batch_size, channels, height, width]
            
        Returns:
            encoded_features: [batch_size, hidden_size]
        """
        features = self.image_encoder(images)
        return self.dropout(self.projections['image'](features))
    
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        前向传播，编码所有可用的输入模态
        
        Args:
            inputs: Dictionary containing different modality inputs
            
        Returns:
            encoded_features: Dictionary of encoded features for each modality
        """
        features = {}
        
        # 编码SMILES（如果提供）
        if 'smiles' in inputs:
            features['smiles'] = self.encode_smiles(
                inputs['smiles']['input_ids'],
                inputs['smiles']['attention_mask']
            )
        
        # 编码文本（如果提供）
        if 'text' in inputs:
            features['text'] = self.encode_text(
                inputs['text']['input_ids'],
                inputs['text']['attention_mask']
            )
        
        # 编码图结构（如果提供）
        if 'graph' in inputs:
            features['graph'] = self.encode_graph(
                inputs['graph'].x,
                inputs['graph'].edge_index,
                inputs['graph'].batch
            )
        
        # 编码图像（如果提供）
        if 'image' in inputs:
            features['image'] = self.encode_image(inputs['image'])
        
        return features
    
    def get_output_dim(self) -> int:
        """返回编码器输出维度"""
        return self.hidden_size


class ModalitySpecificEncoder(nn.Module):
    """单个模态的专用编码器基类"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim 
        self.output_dim = output_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
        

class AdaptivePooling(nn.Module):
    """自适应池化层，支持多种池化策略"""
    
    def __init__(self, pooling_type: str = 'attention'):
        super().__init__()
        self.pooling_type = pooling_type
        
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
        """
        if self.pooling_type == 'mean':
            if attention_mask is not None:
                return (hidden_states * attention_mask.unsqueeze(-1)).sum(1) / \
                       attention_mask.sum(-1, keepdim=True).clamp(min=1)
            else:
                return hidden_states.mean(1)
                
        elif self.pooling_type == 'max':
            return hidden_states.max(1)[0]
            
        elif self.pooling_type == 'cls':
            return hidden_states[:, 0]  # 第一个token (通常是[CLS])
            
        elif self.pooling_type == 'attention':
            # 可学习的注意力池化
            attention_weights = torch.softmax(
                hidden_states.sum(-1), dim=1
            ).unsqueeze(-1)
            return (hidden_states * attention_weights).sum(1)
            
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")