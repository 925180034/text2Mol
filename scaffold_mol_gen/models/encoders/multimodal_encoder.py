"""
统一的多模态编码器
整合所有模态的编码器，提供统一接口
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union
import logging
from pathlib import Path

from .smiles_encoder import BartSMILESEncoder
from .text_encoder import BERTEncoder, SciBERTEncoder
from .graph_encoder import GINEncoder, GraphFeatureExtractor
from .image_encoder import SwinTransformerEncoder, MolecularImageGenerator

logger = logging.getLogger(__name__)

class MultiModalEncoder(nn.Module):
    """
    统一的多模态编码器
    支持Scaffold的三种模态（SMILES/Graph/Image）和文本描述的编码
    """
    
    def __init__(self,
                 hidden_size: int = 768,
                 use_scibert: bool = False,
                 freeze_backbones: bool = False,
                 device: str = 'cuda'):
        """
        初始化多模态编码器
        
        Args:
            hidden_size: 统一的隐藏层维度
            use_scibert: 是否使用SciBERT（否则使用BERT）
            freeze_backbones: 是否冻结预训练模型
            device: 设备
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.device = device
        
        logger.info("初始化多模态编码器...")
        
        # 1. SMILES编码器（用于Scaffold SMILES）
        logger.info("创建SMILES编码器...")
        self.smiles_encoder = BartSMILESEncoder(
            hidden_size=hidden_size,
            freeze_backbone=freeze_backbones
        )
        
        # 2. 文本编码器（用于分子描述）
        logger.info(f"创建文本编码器 ({'SciBERT' if use_scibert else 'BERT'})...")
        if use_scibert:
            self.text_encoder = SciBERTEncoder(
                hidden_size=hidden_size,
                freeze_backbone=freeze_backbones
            )
        else:
            self.text_encoder = BERTEncoder(
                hidden_size=hidden_size,
                freeze_backbone=freeze_backbones
            )
        
        # 3. 图编码器（用于Scaffold Graph）
        logger.info("创建GIN图编码器...")
        self.graph_encoder = GINEncoder(
            input_dim=9,  # 原子特征维度（实际是9维）
            hidden_dim=hidden_size,
            output_dim=hidden_size,
            num_layers=5
        )
        self.graph_extractor = GraphFeatureExtractor()
        
        # 4. 图像编码器（用于Scaffold Image）
        logger.info("创建Swin Transformer图像编码器...")
        self.image_encoder = SwinTransformerEncoder(
            hidden_size=hidden_size,
            freeze_backbone=freeze_backbones
        )
        self.image_generator = MolecularImageGenerator()
        
        # 模态类型映射
        self.modality_encoders = {
            'smiles': self.smiles_encoder,
            'text': self.text_encoder,
            'graph': self.graph_encoder,
            'image': self.image_encoder
        }
        
        logger.info("多模态编码器初始化完成")
    
    def encode_scaffold(self, 
                       scaffold_data: Any,
                       scaffold_modality: str) -> torch.Tensor:
        """
        编码Scaffold（支持三种模态）
        
        Args:
            scaffold_data: Scaffold数据
            scaffold_modality: 模态类型 ('smiles', 'graph', 'image')
            
        Returns:
            scaffold_features: 编码后的特征
        """
        if scaffold_modality == 'smiles':
            # SMILES编码
            if isinstance(scaffold_data, str):
                scaffold_data = [scaffold_data]
            
            inputs = self.smiles_encoder.tokenize(scaffold_data)
            # 确保所有tensor都转移到正确的设备上
            inputs = {k: v.to(self.device) if torch.is_tensor(v) else v 
                     for k, v in inputs.items()}
            
            features = self.smiles_encoder(
                inputs['input_ids'],
                inputs.get('attention_mask')
            )
            # 池化
            scaffold_features = self.smiles_encoder.get_pooled_features(
                features, inputs.get('attention_mask')
            )
            
        elif scaffold_modality == 'graph':
            # Graph编码
            if isinstance(scaffold_data, str):
                # 从SMILES生成图
                graphs = self.graph_extractor.batch_smiles_to_graphs([scaffold_data])
            else:
                # 处理不同类型的graph数据
                from torch_geometric.data import Batch, Data
                
                if isinstance(scaffold_data, Batch):
                    # 从Batch对象转换为Data对象列表
                    graphs = scaffold_data.to_data_list()
                elif isinstance(scaffold_data, (list, tuple)):
                    # 如果已经是列表或元组，直接使用
                    graphs = list(scaffold_data)
                elif isinstance(scaffold_data, Data):
                    # 如果是单个Data对象，转换为列表
                    graphs = [scaffold_data]
                else:
                    # 其他情况，尝试直接使用
                    graphs = scaffold_data
            
            scaffold_features = self.graph_encoder.encode_graphs(graphs)
            
        elif scaffold_modality == 'image':
            # 图像模态
            if isinstance(scaffold_data, torch.Tensor):
                # 已经是tensor，直接传入
                scaffold_features = self.image_encoder(scaffold_data)
            else:
                # 将SMILES转换为图像
                from scaffold_mol_gen.data.multimodal_preprocessor import MultiModalPreprocessor
                preprocessor = MultiModalPreprocessor()
                
                # 处理批量数据
                if isinstance(scaffold_data, list):
                    images = []
                    for smiles in scaffold_data:
                        img = preprocessor.smiles_to_image(smiles)
                        if img is not None:
                            images.append(img)
                        else:
                            # 如果转换失败，创建一个空白图像
                            import numpy as np
                            images.append(np.ones((224, 224, 3), dtype=np.uint8) * 255)
                else:
                    img = preprocessor.smiles_to_image(scaffold_data)
                    if img is not None:
                        images = [img]
                    else:
                        import numpy as np
                        images = [np.ones((224, 224, 3), dtype=np.uint8) * 255]
                
                scaffold_features = self.image_encoder.encode_images(images)
            
        else:
            raise ValueError(f"Unknown scaffold modality: {scaffold_modality}")
        
        return scaffold_features
    
    def encode_text(self, text_data: Union[str, List[str]]) -> torch.Tensor:
        """
        编码文本描述
        
        Args:
            text_data: 文本或文本列表
            
        Returns:
            text_features: 编码后的特征
        """
        if isinstance(text_data, str):
            text_data = [text_data]
        
        text_features = self.text_encoder.encode(text_data)
        
        # 如果是序列输出，进行池化
        if len(text_features.shape) == 3:
            text_features = text_features.mean(dim=1)
        
        return text_features
    
    def forward(self,
                scaffold_data: Any,
                text_data: Union[str, List[str]],
                scaffold_modality: str = 'smiles'):
        """
        前向传播
        
        Args:
            scaffold_data: Scaffold数据
            text_data: 文本描述
            scaffold_modality: Scaffold模态类型
            
        Returns:
            scaffold_features: Scaffold特征
            text_features: 文本特征
        """
        # 编码Scaffold
        scaffold_features = self.encode_scaffold(scaffold_data, scaffold_modality)
        
        # 编码文本
        text_features = self.encode_text(text_data)
        
        return scaffold_features, text_features
    
    def get_encoder(self, modality: str):
        """
        获取特定模态的编码器
        
        Args:
            modality: 模态类型
            
        Returns:
            encoder: 对应的编码器
        """
        if modality not in self.modality_encoders:
            raise ValueError(f"Unknown modality: {modality}")
        return self.modality_encoders[modality]
    
    def to(self, device):
        """移动到指定设备"""
        super().to(device)
        self.device = device
        return self