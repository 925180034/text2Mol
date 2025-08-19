"""
分子图像解码器
将768维特征向量解码为分子2D图像
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import logging
from PIL import Image
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

class MolecularImageDecoder(nn.Module):
    """
    分子图像解码器
    将统一的768维特征向量解码为分子2D图像
    使用类似DCGAN的生成器架构
    """
    
    def __init__(self,
                 input_dim: int = 768,
                 image_size: int = 224,
                 channels: int = 3,  # RGB
                 hidden_dim: int = 512,
                 num_layers: int = 4,
                 dropout: float = 0.1):
        """
        Args:
            input_dim: 输入特征维度
            image_size: 输出图像尺寸
            channels: 图像通道数 (3 for RGB)
            hidden_dim: 隐藏层维度
            num_layers: 反卷积层数
            dropout: Dropout率
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.image_size = image_size
        self.channels = channels
        self.hidden_dim = hidden_dim
        
        # 计算初始特征图尺寸
        self.init_size = image_size // (2 ** num_layers)  # 14 for 224x224 with 4 layers
        
        # 输入投影层：768 -> hidden_dim * init_size^2
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 使用LayerNorm避免批次大小问题
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * self.init_size * self.init_size),
            nn.LayerNorm(hidden_dim * self.init_size * self.init_size),  # 使用LayerNorm
            nn.ReLU(inplace=True)
        )
        
        # 反卷积层序列
        layers = []
        in_channels = hidden_dim
        
        # 逐步上采样
        for i in range(num_layers):
            out_channels = hidden_dim // (2 ** (i + 1)) if i < num_layers - 1 else channels
            
            layers.extend([
                nn.ConvTranspose2d(
                    in_channels, 
                    out_channels if i == num_layers - 1 else out_channels,
                    kernel_size=4, 
                    stride=2, 
                    padding=1,
                    bias=False
                ),
            ])
            
            if i < num_layers - 1:  # 不在最后一层添加BN和激活
                layers.extend([
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout)
                ])
            
            in_channels = out_channels
        
        # 最后添加Tanh激活输出[-1, 1]
        layers.append(nn.Tanh())
        
        self.deconv_layers = nn.Sequential(*layers)
        
        # 特征增强网络
        self.feature_enhancer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ) for _ in range(2)
        ])
        
        # 风格控制层（可选）
        self.style_controller = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.Sigmoid()
        )
        
        logger.info(f"初始化分子图像解码器: {input_dim}→{image_size}x{image_size}x{channels}")
        logger.info(f"初始特征图尺寸: {self.init_size}x{self.init_size}")
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: [batch_size, 768] 输入特征
            
        Returns:
            [batch_size, channels, height, width] 生成的图像
        """
        batch_size = features.size(0)
        
        # 特征增强
        enhanced_features = features
        for enhancer in self.feature_enhancer:
            enhanced_features = enhancer(enhanced_features) + enhanced_features
        
        # 投影到初始特征图
        projected = self.input_projection(enhanced_features)  # [batch_size, hidden_dim * init_size^2]
        
        # 重塑为特征图
        feature_map = projected.view(
            batch_size, 
            self.hidden_dim, 
            self.init_size, 
            self.init_size
        )  # [batch_size, hidden_dim, init_size, init_size]
        
        # 通过反卷积层生成图像
        generated_image = self.deconv_layers(feature_map)  # [batch_size, channels, image_size, image_size]
        
        return generated_image
    
    def generate_images(self, 
                       features: torch.Tensor,
                       num_samples: int = 1,
                       temperature: float = 1.0,
                       add_noise: bool = False,
                       noise_scale: float = 0.1) -> List[torch.Tensor]:
        """
        生成分子图像
        
        Args:
            features: [batch_size, 768] 输入特征
            num_samples: 每个输入生成的样本数
            temperature: 采样温度（控制多样性）
            add_noise: 是否添加噪声
            noise_scale: 噪声规模
            
        Returns:
            生成的图像列表
        """
        self.eval()
        
        with torch.no_grad():
            all_samples = []
            
            for _ in range(num_samples):
                # 可选地添加噪声增加多样性
                if add_noise:
                    noise = torch.randn_like(features) * noise_scale
                    noisy_features = features + noise
                else:
                    noisy_features = features
                
                # 温度缩放（实验性）
                if temperature != 1.0:
                    noisy_features = noisy_features / temperature
                
                # 生成图像
                generated = self(noisy_features)
                all_samples.append(generated)
            
            return all_samples
    
    def postprocess_images(self, 
                          images: torch.Tensor,
                          to_pil: bool = False) -> List[Any]:
        """
        后处理生成的图像
        
        Args:
            images: [batch_size, channels, height, width] 
            to_pil: 是否转换为PIL图像
            
        Returns:
            处理后的图像列表
        """
        # 将[-1, 1]范围转换为[0, 1]
        images = (images + 1.0) / 2.0
        images = torch.clamp(images, 0.0, 1.0)
        
        if to_pil:
            # 转换为PIL图像
            pil_images = []
            for i in range(images.size(0)):
                # 转换为numpy
                img_array = images[i].permute(1, 2, 0).cpu().numpy()
                img_array = (img_array * 255).astype(np.uint8)
                
                # 转换为PIL
                if img_array.shape[2] == 3:  # RGB
                    pil_img = Image.fromarray(img_array, 'RGB')
                else:  # 灰度图
                    pil_img = Image.fromarray(img_array.squeeze(), 'L')
                
                pil_images.append(pil_img)
            
            return pil_images
        else:
            # 返回tensor列表
            return [images[i] for i in range(images.size(0))]


class ImageDecoderLoss(nn.Module):
    """分子图像解码器损失函数"""
    
    def __init__(self,
                 reconstruction_weight: float = 1.0,
                 perceptual_weight: float = 0.1,
                 style_weight: float = 0.01,
                 use_perceptual: bool = False):
        super().__init__()
        
        self.reconstruction_weight = reconstruction_weight
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.use_perceptual = use_perceptual
        
        # 基本重建损失 (L1 + L2)
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        
        # 感知损失（可选，需要预训练的VGG网络）
        if use_perceptual:
            try:
                from torchvision.models import vgg16
                self.vgg = vgg16(pretrained=True).features[:16]  # 使用前几层
                for param in self.vgg.parameters():
                    param.requires_grad = False
                self.vgg.eval()
            except:
                logger.warning("无法加载VGG，禁用感知损失")
                self.use_perceptual = False
    
    def forward(self, 
               generated_images: torch.Tensor,
               target_images: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算损失
        
        Args:
            generated_images: [batch_size, channels, height, width] 生成图像
            target_images: [batch_size, channels, height, width] 目标图像
            
        Returns:
            总损失和各部分损失
        """
        losses = {}
        
        # 重建损失
        l1_loss = self.l1_loss(generated_images, target_images)
        l2_loss = self.l2_loss(generated_images, target_images)
        reconstruction_loss = l1_loss + 0.5 * l2_loss
        
        losses['l1'] = l1_loss.item()
        losses['l2'] = l2_loss.item()
        losses['reconstruction'] = reconstruction_loss.item()
        
        total_loss = self.reconstruction_weight * reconstruction_loss
        
        # 感知损失（可选）
        if self.use_perceptual and hasattr(self, 'vgg'):
            try:
                # 确保图像在[0, 1]范围内
                gen_norm = (generated_images + 1.0) / 2.0
                target_norm = (target_images + 1.0) / 2.0
                
                gen_features = self.vgg(gen_norm)
                target_features = self.vgg(target_norm)
                
                perceptual_loss = self.l2_loss(gen_features, target_features)
                losses['perceptual'] = perceptual_loss.item()
                
                total_loss += self.perceptual_weight * perceptual_loss
            except Exception as e:
                logger.warning(f"感知损失计算失败: {e}")
        
        return total_loss, losses


def test_image_decoder():
    """测试分子图像解码器"""
    print("测试分子图像解码器...")
    
    # 创建解码器
    decoder = MolecularImageDecoder(
        input_dim=768,
        image_size=224,
        channels=3,
        hidden_dim=256,
        num_layers=4
    )
    
    # 测试输入
    batch_size = 2
    features = torch.randn(batch_size, 768)
    
    print(f"输入特征形状: {features.shape}")
    
    # 前向传播
    generated_images = decoder(features)
    print(f"生成图像形状: {generated_images.shape}")
    print(f"图像值范围: [{generated_images.min().item():.3f}, {generated_images.max().item():.3f}]")
    
    # 测试多样化生成
    generated_samples = decoder.generate_images(
        features, 
        num_samples=3,
        add_noise=True,
        noise_scale=0.1
    )
    print(f"生成了 {len(generated_samples)} 个样本批次")
    
    # 测试后处理
    processed_images = decoder.postprocess_images(generated_images, to_pil=False)
    print(f"后处理得到 {len(processed_images)} 张图像")
    print(f"后处理后值范围: [{processed_images[0].min().item():.3f}, {processed_images[0].max().item():.3f}]")
    
    # 测试损失函数
    loss_fn = ImageDecoderLoss()
    target_images = torch.randn_like(generated_images)
    total_loss, loss_dict = loss_fn(generated_images, target_images)
    
    print(f"\n损失测试:")
    print(f"  总损失: {total_loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")
    
    print("✅ 分子图像解码器测试完成")


if __name__ == "__main__":
    test_image_decoder()