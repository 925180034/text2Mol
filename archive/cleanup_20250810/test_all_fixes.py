#!/usr/bin/env python3
"""
测试所有修复
验证MolT5生成、Graph输入和Image输入的修复
"""

import sys
import torch
import logging
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

# 导入修复模块
from fix_generation_quality import FixedMolT5Generator
from fix_graph_input import FixedGraphProcessor
from fix_image_input import FixedImageProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def test_all_fixes():
    """测试所有修复"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    
    # 测试数据
    test_smiles = [
        "CCO",  # 乙醇
        "CC(=O)O",  # 乙酸
        "c1ccccc1",  # 苯
    ]
    
    results = {
        'generation': False,
        'graph_input': False,
        'image_input': False
    }
    
    logger.info("\n" + "="*60)
    logger.info("测试修复1: MolT5生成质量")
    logger.info("="*60)
    
    try:
        # 测试生成质量修复
        generator = FixedMolT5Generator(device=device)
        
        # 创建模拟的encoder输出
        batch_size = len(test_smiles)
        seq_len = 32
        hidden_size = 768
        
        encoder_outputs = torch.randn(batch_size, seq_len, hidden_size).to(device)
        attention_mask = torch.ones(batch_size, seq_len).to(device)
        
        # 生成SMILES
        generated_smiles = generator.generate_smiles(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            num_beams=3,
            max_length=128,
            temperature=0.8
        )
        
        logger.info(f"生成了{len(generated_smiles)}个SMILES:")
        for i, smiles in enumerate(generated_smiles):
            logger.info(f"  {i+1}. {smiles[:50]}{'...' if len(smiles) > 50 else ''}")
        
        # 检查是否不再生成随机文本
        bad_words = ['Sand', 'brick', 'hell', 'Cub', 'rock']
        is_text = any(word in str(generated_smiles) for word in bad_words)
        
        if not is_text:
            logger.info("✅ 生成修复成功: 不再生成随机文本")
            results['generation'] = True
        else:
            logger.warning("⚠️ 生成修复部分成功: 仍有一些文本词汇")
            results['generation'] = 'partial'
            
    except Exception as e:
        logger.error(f"❌ 生成修复失败: {e}")
        results['generation'] = False
    
    logger.info("\n" + "="*60)
    logger.info("测试修复2: Graph输入处理")
    logger.info("="*60)
    
    try:
        # 测试Graph输入修复
        batch = FixedGraphProcessor.prepare_graph_batch(test_smiles, device)
        
        if batch is not None and hasattr(batch, 'x'):
            logger.info(f"✅ Graph批处理成功:")
            logger.info(f"  - 批次大小: {batch.num_graphs}")
            logger.info(f"  - 总节点数: {batch.x.shape[0]}")
            logger.info(f"  - 节点特征维度: {batch.x.shape[1]}")
            logger.info(f"  - 批处理属性: {list(batch.keys())}")
            results['graph_input'] = True
        else:
            logger.error("❌ Graph批处理失败")
            results['graph_input'] = False
            
    except Exception as e:
        logger.error(f"❌ Graph输入修复失败: {e}")
        results['graph_input'] = False
    
    logger.info("\n" + "="*60)
    logger.info("测试修复3: Image输入处理")
    logger.info("="*60)
    
    try:
        # 测试Image输入修复
        processor = FixedImageProcessor(image_size=224)
        batch = processor.prepare_image_batch(test_smiles, device)
        
        if batch is not None:
            logger.info(f"✅ Image批处理成功:")
            logger.info(f"  - 批次形状: {batch.shape}")
            logger.info(f"  - 数值范围: [{batch.min().item():.3f}, {batch.max().item():.3f}]")
            logger.info(f"  - 设备: {batch.device}")
            results['image_input'] = True
        else:
            logger.error("❌ Image批处理失败")
            results['image_input'] = False
            
    except Exception as e:
        logger.error(f"❌ Image输入修复失败: {e}")
        results['image_input'] = False
    
    # 总结
    logger.info("\n" + "="*60)
    logger.info("测试总结")
    logger.info("="*60)
    
    total_fixes = 3
    successful_fixes = sum(1 for v in results.values() if v is True)
    partial_fixes = sum(1 for v in results.values() if v == 'partial')
    
    logger.info(f"成功修复: {successful_fixes}/{total_fixes}")
    logger.info(f"部分修复: {partial_fixes}/{total_fixes}")
    
    for fix_name, status in results.items():
        if status is True:
            status_icon = "✅"
            status_text = "成功"
        elif status == 'partial':
            status_icon = "⚠️"
            status_text = "部分成功"
        else:
            status_icon = "❌"
            status_text = "失败"
        
        logger.info(f"  {status_icon} {fix_name}: {status_text}")
    
    return results


if __name__ == "__main__":
    # 运行各个独立的测试
    logger.info("="*70)
    logger.info("运行独立测试")
    logger.info("="*70)
    
    logger.info("\n1. 测试Graph处理器:")
    logger.info("-"*40)
    from fix_graph_input import test_graph_processor
    test_graph_processor()
    
    logger.info("\n2. 测试Image处理器:")
    logger.info("-"*40)
    from fix_image_input import test_image_processor
    test_image_processor()
    
    logger.info("\n3. 测试生成器修复:")
    logger.info("-"*40)
    from fix_generation_quality import test_fixed_generator
    test_fixed_generator()
    
    # 运行集成测试
    logger.info("\n" + "="*70)
    logger.info("运行集成测试")
    logger.info("="*70)
    
    results = test_all_fixes()
    
    # 最终建议
    logger.info("\n" + "="*70)
    logger.info("建议")
    logger.info("="*70)
    
    if all(v for v in results.values()):
        logger.info("✅ 所有修复都已成功！现在可以:")
        logger.info("  1. 将修复集成到主代码中")
        logger.info("  2. 重新运行完整的多模态测试")
        logger.info("  3. 开始训练改进的模型")
    else:
        logger.info("⚠️ 部分修复需要进一步改进:")
        if not results.get('generation'):
            logger.info("  - 生成质量: 需要重新训练模型或调整tokenizer")
        if not results.get('graph_input'):
            logger.info("  - Graph输入: 检查torch_geometric版本兼容性")
        if not results.get('image_input'):
            logger.info("  - Image输入: 检查PIL和torchvision依赖")