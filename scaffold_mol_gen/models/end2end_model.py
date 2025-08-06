"""
端到端的多模态分子生成模型
整合编码器、融合层和MolT5生成器
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path

# 导入所有组件
from .encoders import MultiModalEncoder
from .fusion_simplified import ModalFusionLayer, MultiModalFusionLayer
from .molt5_adapter import MolT5Generator
from .output_decoders import OutputDecoder

logger = logging.getLogger(__name__)


class End2EndMolecularGenerator(nn.Module):
    """
    完整的端到端分子生成模型
    支持多种输入模态组合和输出模态
    """
    
    def __init__(self,
                 hidden_size: int = 768,
                 molt5_path: str = "/root/autodl-tmp/text2Mol-models/MolT5-Large-Caption2SMILES",
                 use_scibert: bool = False,
                 freeze_encoders: bool = True,
                 freeze_molt5: bool = True,
                 fusion_type: str = 'both',
                 device: str = 'cuda'):
        """
        Args:
            hidden_size: 统一的隐藏层维度
            molt5_path: MolT5模型路径
            use_scibert: 是否使用SciBERT
            freeze_encoders: 是否冻结编码器
            freeze_molt5: 是否冻结MolT5
            fusion_type: 融合类型 ('attention', 'gated', 'both')
            device: 设备
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.device = device
        
        logger.info("初始化端到端分子生成模型...")
        
        # 1. 多模态编码器
        logger.info("创建多模态编码器...")
        self.encoder = MultiModalEncoder(
            hidden_size=hidden_size,
            use_scibert=use_scibert,
            freeze_backbones=freeze_encoders,
            device=device
        )
        
        # 2. 模态融合层
        logger.info(f"创建融合层 (type={fusion_type})...")
        self.fusion = MultiModalFusionLayer(
            hidden_size=hidden_size,
            num_heads=8,
            dropout=0.1
        )
        
        # 3. MolT5生成器
        logger.info("创建MolT5生成器...")
        self.generator = MolT5Generator(
            molt5_path=molt5_path,
            adapter_config={
                'input_hidden_size': hidden_size,
                'num_layers': 2,
                'num_heads': 8
            },
            freeze_molt5=freeze_molt5,
            device=device
        )
        
        # 4. 输出解码器
        logger.info("创建输出解码器...")
        self.output_decoder = OutputDecoder()
        self.output_modalities = ['smiles', 'graph', 'image']
        
        logger.info("端到端模型初始化完成")
        
    def forward(self,
                scaffold_data: Any,
                text_data: Union[str, List[str]],
                scaffold_modality: str = 'smiles',
                target_smiles: Optional[List[str]] = None,
                output_modality: str = 'smiles') -> Dict[str, Any]:
        """
        前向传播
        
        Args:
            scaffold_data: Scaffold数据（SMILES字符串、图或图像）
            text_data: 文本描述
            scaffold_modality: Scaffold输入模态 ('smiles', 'graph', 'image')
            target_smiles: 目标SMILES（训练时提供）
            output_modality: 输出模态 ('smiles', 'graph', 'image')
            
        Returns:
            包含生成结果和中间信息的字典
        """
        # 1. 编码输入
        scaffold_features, text_features = self.encoder(
            scaffold_data=scaffold_data,
            text_data=text_data,
            scaffold_modality=scaffold_modality
        )
        
        # 2. 模态融合
        fused_features, fusion_info = self.fusion(
            scaffold_features=scaffold_features,
            text_features=text_features,
            scaffold_modality=scaffold_modality
        )
        
        # 3. 生成输出
        output_dict = {}
        
        if output_modality == 'smiles':
            # SMILES生成
            generator_output = self.generator(
                fused_features=fused_features,
                target_smiles=target_smiles
            )
            output_dict.update(generator_output)
            
        elif output_modality == 'graph':
            # TODO: 实现Graph解码器
            raise NotImplementedError("Graph输出模态尚未实现")
            
        elif output_modality == 'image':
            # TODO: 实现Image解码器
            raise NotImplementedError("Image输出模态尚未实现")
            
        else:
            raise ValueError(f"不支持的输出模态: {output_modality}")
        
        # 4. 添加中间信息
        output_dict['fusion_info'] = fusion_info
        output_dict['scaffold_modality'] = scaffold_modality
        output_dict['output_modality'] = output_modality
        
        return output_dict
    
    def generate(self,
                 scaffold_data: Any,
                 text_data: Union[str, List[str]],
                 scaffold_modality: str = 'smiles',
                 output_modality: str = 'smiles',
                 num_beams: int = 5,
                 temperature: float = 0.8,
                 max_length: int = 128,
                 num_return_sequences: int = 1) -> Union[List[str], Any]:
        """
        生成分子
        
        Args:
            scaffold_data: Scaffold输入
            text_data: 文本描述
            scaffold_modality: Scaffold模态
            output_modality: 输出模态
            num_beams: Beam search大小
            temperature: 采样温度
            max_length: 最大长度
            num_return_sequences: 返回序列数
            
        Returns:
            生成的分子（SMILES/Graph/Image）
        """
        # 1. 编码和融合
        scaffold_features, text_features = self.encoder(
            scaffold_data=scaffold_data,
            text_data=text_data,
            scaffold_modality=scaffold_modality
        )
        
        fused_features, _ = self.fusion(
            scaffold_features=scaffold_features,
            text_features=text_features,
            scaffold_modality=scaffold_modality
        )
        
        # 2. 根据输出模态生成
        # 首先生成SMILES（作为中间表示）
        smiles_output = self.generator.generate(
            fused_features=fused_features,
            num_beams=num_beams,
            temperature=temperature,
            max_length=max_length,
            num_return_sequences=num_return_sequences
        )
        
        # 3. 根据目标模态进行解码
        if output_modality == 'smiles':
            return smiles_output
        else:
            # 使用输出解码器转换到目标模态
            return self.output_decoder.decode(smiles_output, output_modality)
    
    def compute_loss(self,
                     scaffold_data: Any,
                     text_data: Union[str, List[str]],
                     target_smiles: List[str],
                     scaffold_modality: str = 'smiles') -> torch.Tensor:
        """
        计算训练损失
        
        Args:
            scaffold_data: Scaffold输入
            text_data: 文本描述
            target_smiles: 目标SMILES
            scaffold_modality: Scaffold模态
            
        Returns:
            loss: 训练损失
        """
        output_dict = self.forward(
            scaffold_data=scaffold_data,
            text_data=text_data,
            scaffold_modality=scaffold_modality,
            target_smiles=target_smiles,
            output_modality='smiles'
        )
        
        return output_dict['loss']
    
    def get_supported_combinations(self) -> List[Tuple[str, str]]:
        """
        获取支持的输入输出组合
        
        Returns:
            支持的(scaffold_modality, output_modality)组合列表
        """
        scaffold_modalities = ['smiles', 'graph', 'image']
        output_modalities = self.output_modalities
        
        combinations = []
        for scaffold_mod in scaffold_modalities:
            for output_mod in output_modalities:
                combinations.append((scaffold_mod, output_mod))
        
        return combinations


def test_end2end_model():
    """测试端到端模型"""
    import torch
    from rdkit import Chem
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"使用设备: {device}")
    print("=" * 50)
    
    # 检查MolT5模型
    molt5_path = "/root/autodl-tmp/text2Mol-models/MolT5-Large-Caption2SMILES"
    if not Path(molt5_path).exists():
        print(f"⚠️ MolT5模型未找到: {molt5_path}")
        print("请先下载MolT5模型")
        return
    
    # 创建模型
    print("创建端到端模型...")
    model = End2EndMolecularGenerator(
        hidden_size=768,
        molt5_path=molt5_path,
        use_scibert=False,
        freeze_encoders=True,
        freeze_molt5=True,
        fusion_type='both',
        device=device
    )
    
    # 准备测试数据
    print("\n准备测试数据...")
    # Scaffold SMILES (苯环)
    scaffold_smiles = "c1ccccc1"
    # 文本描述
    text_description = "Anti-inflammatory drug with carboxylic acid group"
    
    print(f"Scaffold SMILES: {scaffold_smiles}")
    print(f"文本描述: {text_description}")
    
    # 测试不同的Scaffold输入模态
    print("\n" + "=" * 50)
    print("测试不同的输入模态组合")
    print("=" * 50)
    
    for scaffold_modality in ['smiles', 'graph', 'image']:
        print(f"\n### 测试 Scaffold({scaffold_modality}) + Text → SMILES")
        
        try:
            # 生成分子
            generated_smiles = model.generate(
                scaffold_data=scaffold_smiles,
                text_data=text_description,
                scaffold_modality=scaffold_modality,
                output_modality='smiles',
                num_beams=3,
                temperature=0.8,
                max_length=64,
                num_return_sequences=1
            )
            
            print(f"生成的SMILES: {generated_smiles[0]}")
            
            # 验证生成的SMILES
            mol = Chem.MolFromSmiles(generated_smiles[0])
            if mol is not None:
                print("✅ 生成的SMILES有效")
            else:
                print("⚠️ 生成的SMILES无效")
                
        except Exception as e:
            print(f"❌ 错误: {e}")
    
    # 测试训练模式
    print("\n" + "=" * 50)
    print("测试训练模式")
    print("=" * 50)
    
    # 批量数据
    batch_scaffold = [scaffold_smiles, "c1ccc2c(c1)cccc2"]  # 苯环和萘环
    batch_text = [
        "Anti-inflammatory drug",
        "Antibiotic compound"
    ]
    batch_target = ["CC(=O)Oc1ccccc1C(=O)O", "CC1CC(C)CN1C(=O)C"]  # 阿司匹林和其他
    
    try:
        loss = model.compute_loss(
            scaffold_data=batch_scaffold,
            text_data=batch_text,
            target_smiles=batch_target,
            scaffold_modality='smiles'
        )
        
        print(f"训练Loss: {loss.item():.4f}")
        print("✅ 训练模式测试通过")
        
    except Exception as e:
        print(f"❌ 训练模式错误: {e}")
    
    # 显示支持的组合
    print("\n" + "=" * 50)
    print("支持的输入输出组合")
    print("=" * 50)
    
    combinations = model.get_supported_combinations()
    for i, (scaffold_mod, output_mod) in enumerate(combinations, 1):
        status = "✅" if output_mod == 'smiles' else "🔄"
        print(f"{i}. Scaffold({scaffold_mod}) + Text → {output_mod} {status}")
    
    print("\n✅ - 已实现  🔄 - 开发中")
    
    print("\n" + "=" * 50)
    print("测试完成！")
    print("=" * 50)


if __name__ == "__main__":
    test_end2end_model()