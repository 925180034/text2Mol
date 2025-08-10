#!/usr/bin/env python3
"""
使用正确的MolT5-Large-Caption2SMILES模型测试
"""

import os
import sys
import torch
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入组件
from scaffold_mol_gen.models.end2end_model import End2EndMolecularGenerator
from transformers import T5ForConditionalGeneration, T5Tokenizer

def main():
    logger.info("="*70)
    logger.info("使用正确的MolT5-Large-Caption2SMILES模型测试")
    logger.info("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建新模型，使用正确的MolT5
    molt5_path = "/root/autodl-tmp/text2Mol-models/MolT5-Large-Caption2SMILES"
    
    if not Path(molt5_path).exists():
        logger.error(f"MolT5-Large模型未找到: {molt5_path}")
        logger.error("请确保下载了正确的模型")
        return
    
    logger.info(f"加载正确的MolT5模型: {molt5_path}")
    
    # 创建模型
    model = End2EndMolecularGenerator(
        hidden_size=768,
        molt5_path=molt5_path,  # 使用正确的MolT5模型
        device=str(device)
    )
    
    # 替换MolT5组件为正确的版本
    logger.info("替换为MolT5-Large-Caption2SMILES...")
    molt5_model = T5ForConditionalGeneration.from_pretrained(molt5_path)
    molt5_tokenizer = T5Tokenizer.from_pretrained(molt5_path)
    
    model.generator.molt5 = molt5_model
    model.generator.tokenizer = molt5_tokenizer
    
    # 如果有训练的checkpoint，加载权重（但不加载错误的MolT5部分）
    checkpoint_path = "/root/autodl-tmp/text2Mol-outputs/optimized_20250809_105726/best_model.pt"
    if Path(checkpoint_path).exists():
        logger.info(f"加载训练的权重（除MolT5外）: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 只加载非MolT5部分的权重
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # 过滤掉MolT5的权重
            filtered_state_dict = {k: v for k, v in state_dict.items() 
                                  if 'molt5' not in k.lower()}
            model.load_state_dict(filtered_state_dict, strict=False)
            logger.info("已加载训练的编码器和融合层权重")
    
    model.to(device)
    model.eval()
    
    # 准备测试数据
    test_df = pd.read_csv("Datasets/test.csv").head(5)
    test_smiles = test_df['SMILES'].tolist()
    test_texts = test_df['description'].tolist() if 'description' in test_df else test_df['text'].tolist()
    
    logger.info(f"测试样本数: {len(test_smiles)}")
    
    # 测试SMILES → SMILES
    logger.info("\n" + "="*60)
    logger.info("测试 SMILES → SMILES (使用正确的MolT5)")
    logger.info("="*60)
    
    try:
        with torch.no_grad():
            generated = model.generate(
                scaffold_data=test_smiles,
                text_data=test_texts,
                scaffold_modality='smiles',
                output_modality='smiles',
                num_beams=5,
                temperature=1.0,
                max_length=128
            )
        
        # 评估
        from rdkit import Chem
        valid_count = 0
        
        logger.info("\n生成结果:")
        for i, (input_s, gen_s) in enumerate(zip(test_smiles[:3], generated[:3])):
            mol = Chem.MolFromSmiles(gen_s)
            is_valid = mol is not None
            valid_count += is_valid
            
            logger.info(f"\n样本 {i+1}:")
            logger.info(f"  输入SMILES: {input_s[:50]}...")
            logger.info(f"  输入文本: {test_texts[i][:80]}...")
            logger.info(f"  生成SMILES: {gen_s}")
            logger.info(f"  有效性: {'✅ 有效' if is_valid else '❌ 无效'}")
        
        validity = sum(1 for s in generated if Chem.MolFromSmiles(s) is not None) / len(generated)
        uniqueness = len(set(generated)) / len(generated)
        
        logger.info(f"\n📊 统计:")
        logger.info(f"  有效率: {validity:.2%}")
        logger.info(f"  唯一性: {uniqueness:.2%}")
        
        if validity > 0:
            logger.info("\n🎉 成功！使用正确的MolT5模型可以生成有效的SMILES！")
        else:
            logger.info("\n⚠️ 生成质量仍然较差，可能需要微调MolT5模型")
            
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("\n" + "="*70)
    logger.info("测试完成")
    logger.info("="*70)

if __name__ == "__main__":
    main()