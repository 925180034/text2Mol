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
                 molt5_path: str = "/root/autodl-tmp/text2Mol-models/molt5-base",
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
        
        # 4. 输出解码器（用于SMILES→其他模态转换）
        logger.info("创建输出解码器...")
        self.output_decoder = OutputDecoder()
        self.output_modalities = ['smiles', 'graph', 'image']
        
        # 5. 直接解码器（用于768维特征→其他模态）
        from .graph_decoder import MolecularGraphDecoder
        from .image_decoder import MolecularImageDecoder
        
        self.direct_graph_decoder = MolecularGraphDecoder(
            input_dim=hidden_size,
            max_atoms=100,
            hidden_dim=512,
            dropout=0.1
        ).to(device)
        
        self.direct_image_decoder = MolecularImageDecoder(
            input_dim=hidden_size,
            image_size=224,
            channels=3,
            hidden_dim=512,
            dropout=0.1
        ).to(device)
        
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
        
        # 首先总是生成SMILES（作为基础表示）
        generator_output = self.generator(
            fused_features=fused_features,
            target_smiles=target_smiles
        )
        output_dict.update(generator_output)
        
        # 如果需要其他模态，支持两种方式：直接解码或转换SMILES
        if output_modality != 'smiles':
            # 方式1：直接从融合特征生成（训练时使用）
            if output_modality == 'graph':
                direct_graph = self.direct_graph_decoder(fused_features)
                output_dict['direct_graph_predictions'] = direct_graph
            elif output_modality == 'image':
                direct_image = self.direct_image_decoder(fused_features)
                output_dict['direct_image'] = direct_image
            
            # 方式2：转换SMILES（推理时使用）
            if 'generated_smiles' in output_dict:
                # 推理模式：转换生成的SMILES
                decoded_output = self.output_decoder.decode(
                    output_dict['generated_smiles'], 
                    output_modality
                )
                output_dict[f'generated_{output_modality}'] = decoded_output
            elif target_smiles is not None:
                # 训练模式：转换目标SMILES用于可视化
                decoded_output = self.output_decoder.decode(
                    target_smiles, 
                    output_modality
                )
                output_dict[f'target_{output_modality}'] = decoded_output
        
        # 4. 添加中间信息
        output_dict['fusion_info'] = fusion_info
        output_dict['scaffold_modality'] = scaffold_modality
        output_dict['output_modality'] = output_modality
        output_dict['fused_features'] = fused_features  # 添加融合特征供训练使用
        
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
        
        # 3. 根据目标模态进行生成/解码
        if output_modality == 'smiles':
            return smiles_output
        elif output_modality == 'graph':
            # 直接从融合特征生成图
            graph_predictions = self.direct_graph_decoder(fused_features)
            decoded_graphs = self.direct_graph_decoder.decode_to_graphs(graph_predictions)
            return decoded_graphs
        elif output_modality == 'image':
            # 直接从融合特征生成图像
            generated_images = self.direct_image_decoder(fused_features)
            processed_images = self.direct_image_decoder.postprocess_images(generated_images, to_pil=True)
            return processed_images
        else:
            # 备用方案：通过SMILES转换
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
    molt5_path = "/root/autodl-tmp/text2Mol-models/molt5-base"
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