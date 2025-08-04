#!/usr/bin/env python3
"""
测试多模态编码器
"""

import os
import sys
import torch
import logging
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from scaffold_mol_gen.models.encoders import (
    MultiModalEncoder,
    SMILESEncoder,
    BERTEncoder,
    GINEncoder,
    SwinTransformerEncoder
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_individual_encoders():
    """测试各个独立的编码器"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    
    # 测试数据
    scaffold_smiles = "c1ccc2c(c1)oc1ccccc12"  # dibenzofuran scaffold
    text_description = "A molecule with anti-inflammatory properties"
    
    logger.info("="*60)
    logger.info("测试独立编码器")
    logger.info("="*60)
    
    # 1. 测试SMILES编码器
    logger.info("\n1. 测试SMILES编码器...")
    try:
        smiles_encoder = SMILESEncoder(model_type="molt5").to(device)
        smiles_features = smiles_encoder.encode([scaffold_smiles])
        logger.info(f"✅ SMILES编码器输出形状: {smiles_features.shape}")
    except Exception as e:
        logger.error(f"❌ SMILES编码器测试失败: {e}")
    
    # 2. 测试文本编码器
    logger.info("\n2. 测试BERT文本编码器...")
    try:
        text_encoder = BERTEncoder().to(device)
        text_features = text_encoder.encode([text_description])
        logger.info(f"✅ 文本编码器输出形状: {text_features.shape}")
    except Exception as e:
        logger.error(f"❌ 文本编码器测试失败: {e}")
    
    # 3. 测试图编码器
    logger.info("\n3. 测试GIN图编码器...")
    try:
        from scaffold_mol_gen.models.encoders import GraphFeatureExtractor
        
        graph_encoder = GINEncoder().to(device)
        extractor = GraphFeatureExtractor()
        
        # 将SMILES转换为图
        graphs = extractor.batch_smiles_to_graphs([scaffold_smiles])
        graph_features = graph_encoder.encode_graphs(graphs)
        logger.info(f"✅ 图编码器输出形状: {graph_features.shape}")
    except Exception as e:
        logger.error(f"❌ 图编码器测试失败: {e}")
    
    # 4. 测试图像编码器
    logger.info("\n4. 测试Swin Transformer图像编码器...")
    try:
        from scaffold_mol_gen.models.encoders import MolecularImageGenerator
        
        image_encoder = SwinTransformerEncoder().to(device)
        generator = MolecularImageGenerator()
        
        # 将SMILES转换为图像
        images = generator.batch_smiles_to_images([scaffold_smiles])
        image_features = image_encoder.encode_images(images)
        logger.info(f"✅ 图像编码器输出形状: {image_features.shape}")
    except Exception as e:
        logger.error(f"❌ 图像编码器测试失败: {e}")

def test_multimodal_encoder():
    """测试统一的多模态编码器"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info("\n" + "="*60)
    logger.info("测试多模态编码器")
    logger.info("="*60)
    
    # 测试数据
    scaffold_smiles = "c1ccc2c(c1)oc1ccccc12"
    text_description = "A molecule with anti-inflammatory properties"
    
    try:
        # 创建多模态编码器
        logger.info("\n创建多模态编码器...")
        encoder = MultiModalEncoder(
            hidden_size=768,
            use_scibert=False,
            freeze_backbones=True,
            device=device
        ).to(device)
        
        logger.info("\n测试不同的输入模态组合...")
        
        # 1. SMILES + Text
        logger.info("\n1. Scaffold(SMILES) + Text:")
        scaffold_feat, text_feat = encoder(
            scaffold_data=scaffold_smiles,
            text_data=text_description,
            scaffold_modality='smiles'
        )
        logger.info(f"   Scaffold特征: {scaffold_feat.shape}")
        logger.info(f"   文本特征: {text_feat.shape}")
        
        # 2. Graph + Text
        logger.info("\n2. Scaffold(Graph) + Text:")
        scaffold_feat, text_feat = encoder(
            scaffold_data=scaffold_smiles,  # 会自动转换为图
            text_data=text_description,
            scaffold_modality='graph'
        )
        logger.info(f"   Scaffold特征: {scaffold_feat.shape}")
        logger.info(f"   文本特征: {text_feat.shape}")
        
        # 3. Image + Text
        logger.info("\n3. Scaffold(Image) + Text:")
        scaffold_feat, text_feat = encoder(
            scaffold_data=scaffold_smiles,  # 会自动转换为图像
            text_data=text_description,
            scaffold_modality='image'
        )
        logger.info(f"   Scaffold特征: {scaffold_feat.shape}")
        logger.info(f"   文本特征: {text_feat.shape}")
        
        logger.info("\n✅ 多模态编码器测试成功！")
        
    except Exception as e:
        logger.error(f"\n❌ 多模态编码器测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_batch_processing():
    """测试批处理"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info("\n" + "="*60)
    logger.info("测试批处理")
    logger.info("="*60)
    
    # 批量数据
    scaffold_smiles_list = [
        "c1ccc2c(c1)oc1ccccc12",
        "c1ccc2c(c1)[nH]c1ccccc12",
        "c1ccc(cc1)C(=O)O"
    ]
    text_list = [
        "Anti-inflammatory drug",
        "Serotonin receptor agonist",
        "Analgesic compound"
    ]
    
    try:
        encoder = MultiModalEncoder(device=device).to(device)
        
        logger.info(f"\n处理批量数据 (batch_size={len(scaffold_smiles_list)})...")
        
        # 批量编码SMILES
        scaffold_features = []
        for smiles in scaffold_smiles_list:
            feat = encoder.encode_scaffold(smiles, 'smiles')
            scaffold_features.append(feat)
        scaffold_features = torch.cat(scaffold_features, dim=0)
        
        # 批量编码文本
        text_features = encoder.encode_text(text_list)
        
        logger.info(f"批量Scaffold特征: {scaffold_features.shape}")
        logger.info(f"批量文本特征: {text_features.shape}")
        
        logger.info("\n✅ 批处理测试成功！")
        
    except Exception as e:
        logger.error(f"\n❌ 批处理测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主测试函数"""
    
    logger.info("="*60)
    logger.info("多模态编码器测试")
    logger.info("="*60)
    
    # 1. 测试独立编码器
    test_individual_encoders()
    
    # 2. 测试多模态编码器
    test_multimodal_encoder()
    
    # 3. 测试批处理
    test_batch_processing()
    
    logger.info("\n" + "="*60)
    logger.info("测试完成")
    logger.info("="*60)

if __name__ == "__main__":
    main()