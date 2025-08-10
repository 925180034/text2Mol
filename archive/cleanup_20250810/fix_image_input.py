#!/usr/bin/env python3
"""
修复Image输入处理问题
解决图像预处理和批处理问题
"""

import torch
import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw
import torchvision.transforms as transforms
from typing import List, Optional, Union
import logging
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedImageProcessor:
    """修复的图像处理器"""
    
    def __init__(self, image_size: int = 224):
        """
        初始化图像处理器
        
        Args:
            image_size: 目标图像大小
        """
        self.image_size = image_size
        
        # 定义图像转换
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # 用于处理numpy数组的转换
        self.numpy_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def smiles_to_image(self, smiles: str, size: tuple = None) -> Optional[np.ndarray]:
        """
        将SMILES转换为分子图像
        
        Args:
            smiles: SMILES字符串
            size: 图像大小
            
        Returns:
            图像数组或None
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"无效的SMILES: {smiles}")
                return None
            
            # 生成图像
            size = size or (self.image_size, self.image_size)
            img = Draw.MolToImage(mol, size=size)
            
            # 转换为numpy数组
            img_array = np.array(img)
            
            # 确保是RGB格式
            if len(img_array.shape) == 2:
                # 灰度图转RGB
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[-1] == 4:
                # RGBA转RGB
                img_array = img_array[:, :, :3]
            
            return img_array
            
        except Exception as e:
            logger.error(f"SMILES到图像转换失败 ({smiles}): {e}")
            return None
    
    def create_default_image(self) -> np.ndarray:
        """创建默认的白色图像"""
        return np.ones((self.image_size, self.image_size, 3), dtype=np.uint8) * 255
    
    def process_image(self, image: Union[np.ndarray, Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        处理各种格式的图像输入
        
        Args:
            image: 图像（numpy数组、PIL图像或tensor）
            
        Returns:
            处理后的tensor
        """
        try:
            if isinstance(image, torch.Tensor):
                # 已经是tensor
                if image.dim() == 3:
                    return image
                elif image.dim() == 2:
                    # 灰度图，添加通道维度
                    return image.unsqueeze(0).repeat(3, 1, 1)
                else:
                    raise ValueError(f"不支持的tensor维度: {image.dim()}")
            
            elif isinstance(image, np.ndarray):
                # numpy数组
                if image.dtype != np.uint8:
                    # 归一化到0-255
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                
                # 使用transform处理
                return self.numpy_transform(image)
            
            elif isinstance(image, Image.Image):
                # PIL图像
                return self.transform(image)
            
            else:
                raise ValueError(f"不支持的图像类型: {type(image)}")
                
        except Exception as e:
            logger.error(f"图像处理失败: {e}")
            # 返回默认图像
            default_img = self.create_default_image()
            return self.numpy_transform(default_img)
    
    def prepare_image_batch(self, 
                          smiles_list: List[str], 
                          device: str = 'cuda') -> Optional[torch.Tensor]:
        """
        准备图像批处理数据
        
        Args:
            smiles_list: SMILES字符串列表
            device: 设备
            
        Returns:
            批处理的图像tensor
        """
        images = []
        
        for smiles in smiles_list:
            # 将SMILES转换为图像
            img_array = self.smiles_to_image(smiles)
            
            if img_array is None:
                # 使用默认图像
                logger.warning(f"使用默认图像替代无效的SMILES: {smiles}")
                img_array = self.create_default_image()
            
            # 处理图像
            img_tensor = self.process_image(img_array)
            images.append(img_tensor)
        
        if not images:
            logger.error("没有有效的图像数据")
            return None
        
        # 堆叠成批次
        try:
            batch = torch.stack(images)
            batch = batch.to(device)
            
            logger.info(f"成功创建图像批处理: {batch.shape}")
            return batch
            
        except Exception as e:
            logger.error(f"图像批处理失败: {e}")
            return None
    
    def process_image_input(self, 
                          input_data: Union[List[str], List[np.ndarray], List[Image.Image], torch.Tensor],
                          device: str = 'cuda') -> Optional[torch.Tensor]:
        """
        处理各种格式的图像输入
        
        Args:
            input_data: SMILES列表、图像数组列表、PIL图像列表或tensor
            device: 设备
            
        Returns:
            批处理的图像tensor
        """
        if isinstance(input_data, torch.Tensor):
            # 已经是tensor
            if input_data.dim() == 4:
                # 批次tensor [B, C, H, W]
                return input_data.to(device)
            elif input_data.dim() == 3:
                # 单个图像 [C, H, W]
                return input_data.unsqueeze(0).to(device)
            else:
                logger.error(f"不支持的tensor维度: {input_data.dim()}")
                return None
        
        elif isinstance(input_data, list):
            if len(input_data) == 0:
                logger.error("输入列表为空")
                return None
            
            if isinstance(input_data[0], str):
                # SMILES列表
                return self.prepare_image_batch(input_data, device)
            
            elif isinstance(input_data[0], (np.ndarray, Image.Image)):
                # 图像列表
                try:
                    images = [self.process_image(img) for img in input_data]
                    batch = torch.stack(images)
                    return batch.to(device)
                except Exception as e:
                    logger.error(f"图像列表处理失败: {e}")
                    return None
            
            else:
                logger.error(f"不支持的列表元素类型: {type(input_data[0])}")
                return None
        
        else:
            logger.error(f"不支持的输入类型: {type(input_data)}")
            return None


def test_image_processor():
    """测试图像处理器"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    
    processor = FixedImageProcessor(image_size=224)
    
    # 测试SMILES
    test_smiles = [
        "CCO",  # 乙醇
        "CC(=O)O",  # 乙酸
        "c1ccccc1",  # 苯
        "CC(C)CC(C)(C)O",  # 复杂分子
        "INVALID",  # 无效的SMILES
    ]
    
    logger.info(f"\n测试{len(test_smiles)}个SMILES...")
    
    # 测试单个转换
    logger.info("\n1. 测试单个SMILES转换:")
    for smiles in test_smiles[:3]:
        img_array = processor.smiles_to_image(smiles)
        if img_array is not None:
            logger.info(f"  {smiles}: 图像形状={img_array.shape}, 范围=[{img_array.min()}, {img_array.max()}]")
        else:
            logger.info(f"  {smiles}: 转换失败")
    
    # 测试批处理
    logger.info("\n2. 测试批处理:")
    batch = processor.prepare_image_batch(test_smiles, device)
    if batch is not None:
        logger.info(f"  批处理成功: 形状={batch.shape}, 设备={batch.device}")
        logger.info(f"  数值范围: [{batch.min().item():.3f}, {batch.max().item():.3f}]")
    else:
        logger.info("  批处理失败")
    
    # 测试不同输入格式
    logger.info("\n3. 测试不同输入格式:")
    
    # 测试SMILES列表
    result1 = processor.process_image_input(test_smiles[:3], device)
    logger.info(f"  SMILES列表: {'成功' if result1 is not None else '失败'}")
    
    # 测试numpy数组列表
    arrays = [processor.smiles_to_image(s) for s in test_smiles[:3]]
    arrays = [a for a in arrays if a is not None]
    if arrays:
        result2 = processor.process_image_input(arrays, device)
        logger.info(f"  numpy数组列表: {'成功' if result2 is not None else '失败'}")
    
    # 测试已处理的tensor
    if batch is not None:
        result3 = processor.process_image_input(batch, device)
        logger.info(f"  已批处理tensor: {'成功' if result3 is not None else '失败'}")
    
    # 测试PIL图像
    logger.info("\n4. 测试PIL图像:")
    pil_images = []
    for smiles in test_smiles[:3]:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            img = Draw.MolToImage(mol, size=(224, 224))
            pil_images.append(img)
    
    if pil_images:
        result4 = processor.process_image_input(pil_images, device)
        logger.info(f"  PIL图像列表: {'成功' if result4 is not None else '失败'}")
    
    logger.info("\n测试完成!")
    
    return batch


if __name__ == "__main__":
    test_image_processor()