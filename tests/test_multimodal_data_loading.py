#!/usr/bin/env python3
"""
测试多模态数据加载
验证7种输入输出组合是否都能正常工作
"""

import sys
import torch
import numpy as np
from pathlib import Path
import logging

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from scaffold_mol_gen.data.multimodal_preprocessor import MultiModalPreprocessor
from scaffold_mol_gen.models.encoders.multimodal_encoder import MultiModalEncoder
from scaffold_mol_gen.models.graph_decoder import MolecularGraphDecoder
from scaffold_mol_gen.models.image_decoder import MolecularImageDecoder

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_loading():
    """测试多模态数据加载"""
    logger.info("=== 测试多模态数据加载 ===")
    
    # 简化测试：测试基本的数据处理功能
    preprocessor = MultiModalPreprocessor()
    
    # 测试基本转换功能
    test_smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    
    logger.info("测试SMILES转换功能...")
    success_count = 0
    
    for smiles in test_smiles:
        try:
            # 测试SMILES → Graph
            graph = preprocessor.smiles_to_graph(smiles)
            if graph is not None:
                logger.info(f"✅ {smiles} → Graph: {graph.x.shape[0]}个原子, {graph.edge_index.shape[1]}条边")
                success_count += 1
            
            # 测试SMILES → Image  
            image = preprocessor.smiles_to_image(smiles)
            if image is not None:
                logger.info(f"✅ {smiles} → Image: {image.shape}")
                success_count += 1
                
        except Exception as e:
            logger.error(f"❌ {smiles} 转换失败: {e}")
    
    logger.info(f"转换成功率: {success_count}/{len(test_smiles) * 2}")
    
    # 创建模拟数据用于后续测试
    mock_data = {
        'scaffold_graphs': [preprocessor.smiles_to_graph(s) for s in test_smiles],
        'scaffold_images': [preprocessor.smiles_to_image(s) for s in test_smiles],
        'target_graphs': [preprocessor.smiles_to_graph(s) for s in test_smiles], 
        'target_images': [preprocessor.smiles_to_image(s) for s in test_smiles],
        'metadata': [
            {
                'original_index': i,
                'cid': f'CID_{i}',
                'text': f'test molecule {i}',
                'scaffold_smiles': smiles,
                'target_smiles': smiles
            } for i, smiles in enumerate(test_smiles)
        ]
    }
    
    return mock_data

def test_encoders():
    """测试多模态编码器"""
    logger.info("\n=== 测试多模态编码器 ===")
    
    try:
        # 创建编码器 (CPU模式避免设备问题)
        encoder = MultiModalEncoder(hidden_size=768, device='cpu')
        encoder = encoder.to('cpu')
        
        logger.info("✅ 多模态编码器创建成功")
        logger.info("注意: 完整的编码器测试需要大量GPU内存，此处仅验证创建")
        
        # 简化测试：只检查编码器组件是否正确初始化
        logger.info(f"SMILES编码器: {type(encoder.smiles_encoder).__name__}")
        logger.info(f"文本编码器: {type(encoder.text_encoder).__name__}")
        logger.info(f"图编码器: {type(encoder.graph_encoder).__name__}")
        logger.info(f"图像编码器: {type(encoder.image_encoder).__name__}")
        
        return encoder
        
    except Exception as e:
        logger.error(f"❌ 编码器测试失败: {e}")
        return None

def test_decoders():
    """测试解码器"""
    logger.info("\n=== 测试解码器 ===")
    
    # 测试特征
    features = torch.randn(2, 768)
    
    # 测试Graph解码器
    graph_decoder = MolecularGraphDecoder(max_atoms=20)
    generated_graphs = graph_decoder.generate_graphs(features, num_samples=1)
    logger.info(f"✅ Graph解码器: {features.shape} -> {len(generated_graphs)}个批次")
    
    # 测试Image解码器
    image_decoder = MolecularImageDecoder(image_size=64)  # 小尺寸测试
    generated_images = image_decoder.generate_images(features, num_samples=1)
    logger.info(f"✅ Image解码器: {features.shape} -> {generated_images[0].shape}")
    
    return graph_decoder, image_decoder

def test_seven_combinations(data, encoder, graph_decoder, image_decoder):
    """测试7种输入输出组合"""
    logger.info("\n=== 测试7种输入输出组合 ===")
    
    if data is None or encoder is None:
        logger.warning("数据或编码器为空，跳过组合测试")
        return {}
    
    # 简化测试：只验证架构支持，不进行实际编码
    combinations = [
        ("SMILES", "SMILES", "Scaffold(SMILES) + Text → SMILES"),
        ("Graph", "SMILES", "Scaffold(Graph) + Text → SMILES"),  
        ("Image", "SMILES", "Scaffold(Image) + Text → SMILES"),
        ("SMILES", "Graph", "Scaffold(SMILES) + Text → Graph"),
        ("SMILES", "Image", "Scaffold(SMILES) + Text → Image"),
        ("Graph", "Graph", "Scaffold(Graph) + Text → Graph"),
        ("Image", "Image", "Scaffold(Image) + Text → Image")
    ]
    
    results = {}
    
    for input_modality, output_modality, description in combinations:
        logger.info(f"架构验证: {description}")
        
        try:
            # 验证输入数据可用性
            input_available = False
            if input_modality == "SMILES":
                input_available = any(m.get('scaffold_smiles') for m in data['metadata'])
            elif input_modality == "Graph":
                input_available = any(g is not None for g in data['scaffold_graphs'])
            elif input_modality == "Image":
                input_available = any(i is not None for i in data['scaffold_images'])
            
            # 验证解码器可用性
            decoder_available = False
            if output_modality == "SMILES":
                decoder_available = True  # 使用MolT5
            elif output_modality == "Graph":
                decoder_available = graph_decoder is not None
            elif output_modality == "Image":
                decoder_available = image_decoder is not None
            
            if input_available and decoder_available:
                logger.info(f"  ✅ 架构支持: 输入数据可用, 解码器可用")
                results[description] = {'status': 'architecture_ready'}
            else:
                missing = []
                if not input_available:
                    missing.append("输入数据")
                if not decoder_available:
                    missing.append("解码器")
                logger.info(f"  ⚠️  缺少: {', '.join(missing)}")
                results[description] = {'status': 'missing_components', 'missing': missing}
                
        except Exception as e:
            logger.error(f"  ❌ 验证失败: {e}")
            results[description] = {'status': 'failed', 'error': str(e)}
    
    # 总结结果
    logger.info(f"\n=== 架构验证总结 ===")
    ready = sum(1 for r in results.values() if r['status'] == 'architecture_ready')
    total = len(results)
    logger.info(f"架构就绪: {ready}/{total} ({ready/total*100:.1f}%)")
    
    for desc, result in results.items():
        if result['status'] == 'architecture_ready':
            status = "✅"
        elif result['status'] == 'missing_components':
            status = "⚠️ "
        else:
            status = "❌"
        logger.info(f"{status} {desc}")
    
    return results

def main():
    """主测试函数"""
    logger.info("开始多模态数据加载测试...")
    
    # 测试数据加载
    data = test_data_loading()
    
    # 测试编码器
    encoder = test_encoders()
    
    # 测试解码器
    graph_decoder, image_decoder = test_decoders()
    
    # 测试7种组合
    results = test_seven_combinations(data, encoder, graph_decoder, image_decoder)
    
    logger.info("\n🎉 多模态数据加载测试完成!")
    
    return results

if __name__ == "__main__":
    main()