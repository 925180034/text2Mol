"""
图像编码器模块
使用Swin Transformer处理分子图像
"""

import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
import logging
import os

# 设置离线模式
os.environ['TIMM_MODEL_DIR'] = '/root/autodl-tmp/pretrained_models'
os.environ['HF_HUB_OFFLINE'] = '1'


from typing import List, Optional, Union
import io

logger = logging.getLogger(__name__)

class SwinTransformerEncoder(nn.Module):
    """
    Swin Transformer编码器
    用于编码分子的2D图像表示
    """
    
    def __init__(self,
                 model_name: str = 'swin_base_patch4_window7_224',
                 pretrained: bool = True,
                 hidden_size: int = 768,
                 image_size: int = 224,
                 freeze_backbone: bool = False,
                 num_classes: int = 0):  # 0表示不使用分类头
        """
        初始化Swin Transformer编码器
        
        Args:
            model_name: Swin模型名称
            pretrained: 是否使用预训练权重
            hidden_size: 输出隐藏层维度
            image_size: 输入图像大小
            freeze_backbone: 是否冻结预训练权重
            num_classes: 分类头类别数（0表示不使用）
        """
        super().__init__()
        
        self.model_name = model_name
        self.image_size = image_size
        self.hidden_size = hidden_size
        
        # 加载Swin Transformer
        logger.info(f"加载Swin Transformer: {model_name}")
        self.swin = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=image_size
        )
        
        # 获取Swin的输出维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, image_size, image_size)
            swin_output = self.swin.forward_features(dummy_input)
            
            # 处理不同的输出形状来获取特征维度
            if len(swin_output.shape) == 4:  # [batch, H, W, dim]
                swin_hidden_size = swin_output.shape[-1]  # 最后一个维度是特征维度
            elif len(swin_output.shape) == 3:  # [batch, seq_len, dim]
                swin_hidden_size = swin_output.shape[-1]
            elif len(swin_output.shape) == 2:  # [batch, dim]
                swin_hidden_size = swin_output.shape[1]
            else:
                # 尝试提取最后一个有意义的维度
                swin_hidden_size = swin_output.shape[-1]
        
        logger.info(f"Swin输出维度: {swin_hidden_size}")
        
        # 投影层
        if swin_hidden_size != hidden_size:
            self.projection = nn.Linear(swin_hidden_size, hidden_size)
        else:
            self.projection = nn.Identity()
        
        # 额外的图像特征处理层
        self.image_enhancement = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 图像预处理
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # 冻结backbone
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """冻结预训练模型参数"""
        for param in self.swin.parameters():
            param.requires_grad = False
        logger.info("Swin Transformer backbone已冻结")
    
    def forward(self, images: torch.Tensor):
        """
        前向传播
        
        Args:
            images: 图像tensor [batch_size, 3, H, W]
            
        Returns:
            encoded_features: 编码后的特征
        """
        # 确保输入在正确的设备上
        device = next(self.parameters()).device
        if images.device != device:
            images = images.to(device)
        
        # 通过Swin Transformer提取特征
        features = self.swin.forward_features(images)
        
        # 处理不同的输出形状
        if len(features.shape) == 3:  # [batch, seq_len, dim]
            # 全局平均池化
            features = features.mean(dim=1)
        elif len(features.shape) == 4:  # [batch, H, W, dim]
            # 先调整维度顺序，然后进行全局平均池化
            features = features.permute(0, 3, 1, 2)  # [batch, dim, H, W]
            features = features.mean(dim=[2, 3])  # [batch, dim]
        elif len(features.shape) != 2:
            # 如果不是2D，尝试flatten
            features = features.flatten(1)
        
        # 投影到目标维度
        features = self.projection(features)
        
        # 图像特征增强
        features = self.image_enhancement(features)
        
        return features
    
    def preprocess_image(self, image: Union[Image.Image, np.ndarray, str]) -> torch.Tensor:
        """
        预处理单张图像
        
        Args:
            image: PIL图像、numpy数组或图像路径
            
        Returns:
            preprocessed: 预处理后的图像tensor
        """
        if isinstance(image, str):
            # 从路径加载图像
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # 从numpy数组创建PIL图像
            image = Image.fromarray(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # 应用预处理
        return self.image_transform(image)
    
    def encode_images(self, images: List[Union[Image.Image, np.ndarray, str]]):
        """
        编码图像列表
        
        Args:
            images: 图像列表（PIL图像、numpy数组或路径）
            
        Returns:
            encoded_features: 编码后的特征 [batch_size, hidden_size]
        """
        # 预处理所有图像
        preprocessed = []
        for img in images:
            preprocessed.append(self.preprocess_image(img))
        
        # Stack成batch
        batch = torch.stack(preprocessed)
        
        # Move to device
        device = next(self.parameters()).device
        batch = batch.to(device)
        
        # 编码
        with torch.no_grad():
            encoded_features = self.forward(batch)
        
        return encoded_features


class MolecularImageGenerator:
    """
    分子图像生成器
    将SMILES转换为2D分子图像
    """
    
    @staticmethod
    def smiles_to_image(smiles: str, 
                       image_size: int = 224,
                       mol_size: tuple = (300, 300)) -> Image.Image:
        """
        将SMILES转换为分子图像
        
        Args:
            smiles: SMILES字符串
            image_size: 最终图像大小
            mol_size: 分子渲染大小
            
        Returns:
            image: PIL图像
        """
        from rdkit import Chem
        from rdkit.Chem import Draw
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # 返回空白图像
            return Image.new('RGB', (image_size, image_size), color='white')
        
        # 生成2D坐标
        from rdkit.Chem import AllChem
        AllChem.Compute2DCoords(mol)
        
        # 渲染分子
        img = Draw.MolToImage(mol, size=mol_size)
        
        # 调整大小
        img = img.resize((image_size, image_size), Image.LANCZOS)
        
        return img
    
    @staticmethod
    def smiles_to_image_with_highlight(smiles: str,
                                      scaffold_smiles: str = None,
                                      image_size: int = 224) -> Image.Image:
        """
        生成带有scaffold高亮的分子图像
        
        Args:
            smiles: 完整分子SMILES
            scaffold_smiles: scaffold SMILES（用于高亮）
            image_size: 图像大小
            
        Returns:
            image: PIL图像
        """
        from rdkit import Chem
        from rdkit.Chem import Draw
        from rdkit.Chem import AllChem
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return Image.new('RGB', (image_size, image_size), color='white')
        
        # 生成2D坐标
        AllChem.Compute2DCoords(mol)
        
        # 如果提供了scaffold，进行高亮
        highlight_atoms = []
        if scaffold_smiles:
            scaffold = Chem.MolFromSmiles(scaffold_smiles)
            if scaffold:
                match = mol.GetSubstructMatch(scaffold)
                highlight_atoms = list(match)
        
        # 渲染分子
        drawer = Draw.MolDraw2DCairo(image_size, image_size)
        drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms)
        drawer.FinishDrawing()
        
        # 转换为PIL图像
        img_data = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(img_data))
        
        return img
    
    @staticmethod
    def batch_smiles_to_images(smiles_list: List[str],
                              image_size: int = 224) -> List[Image.Image]:
        """
        批量转换SMILES为图像
        
        Args:
            smiles_list: SMILES字符串列表
            image_size: 图像大小
            
        Returns:
            images: PIL图像列表
        """
        generator = MolecularImageGenerator()
        images = []
        for smiles in smiles_list:
            img = generator.smiles_to_image(smiles, image_size)
            images.append(img)
        return images