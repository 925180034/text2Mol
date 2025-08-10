#!/usr/bin/env python3
"""
使用已训练的模型测试
"""

import os
import sys
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
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
from fix_graph_input import FixedGraphProcessor
from fix_image_input import FixedImageProcessor

# 评估指标
def calculate_validity(smiles_list):
    from rdkit import Chem
    valid = sum(1 for s in smiles_list if Chem.MolFromSmiles(s) is not None)
    return valid / len(smiles_list) if smiles_list else 0

def calculate_uniqueness(smiles_list):
    unique = len(set(smiles_list))
    return unique / len(smiles_list) if smiles_list else 0

def main():
    # 使用您训练好的模型
    model_path = "/root/autodl-tmp/text2Mol-outputs/optimized_20250809_105726/best_model.pt"
    
    logger.info(f"加载训练好的模型: {model_path}")
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    # 从checkpoint恢复模型
    if 'model_state_dict' in checkpoint:
        model = End2EndMolecularGenerator(device=str(device))
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model = checkpoint['model']
    else:
        logger.error("无法从checkpoint加载模型")
        return
    
    model.to(device)
    model.eval()
    
    logger.info(f"模型加载成功，设备: {device}")
    
    # 准备测试数据
    test_df = pd.read_csv("Datasets/test.csv").head(5)
    test_smiles = test_df['SMILES'].tolist()
    test_texts = test_df['description'].tolist() if 'description' in test_df else test_df['text'].tolist()
    
    logger.info(f"测试样本数: {len(test_smiles)}")
    
    # 使用修复的处理器
    graph_processor = FixedGraphProcessor
    image_processor = FixedImageProcessor(image_size=224)
    
    results = {}
    
    # 测试1: SMILES → SMILES
    logger.info("\n" + "="*60)
    logger.info("测试 SMILES → SMILES")
    logger.info("="*60)
    
    try:
        with torch.no_grad():
            generated = model.generate(
                scaffold_data=test_smiles,
                text_data=test_texts,
                scaffold_modality='smiles',
                output_modality='smiles',
                num_beams=5,
                temperature=0.8,
                max_length=128
            )
        
        validity = calculate_validity(generated)
        uniqueness = calculate_uniqueness(generated)
        
        logger.info(f"生成的SMILES示例:")
        for i, (input_s, gen_s) in enumerate(zip(test_smiles[:3], generated[:3])):
            logger.info(f"  输入: {input_s[:50]}...")
            logger.info(f"  生成: {gen_s}")
        
        logger.info(f"有效率: {validity:.2%}")
        logger.info(f"唯一性: {uniqueness:.2%}")
        
        results['smiles_to_smiles'] = {
            'validity': validity,
            'uniqueness': uniqueness,
            'examples': generated[:3]
        }
        
    except Exception as e:
        logger.error(f"SMILES→SMILES失败: {e}")
        results['smiles_to_smiles'] = {'error': str(e)}
    
    # 测试2: IMAGE → SMILES
    logger.info("\n" + "="*60)
    logger.info("测试 IMAGE → SMILES")
    logger.info("="*60)
    
    try:
        # 准备图像批次
        image_batch = image_processor.prepare_image_batch(test_smiles, str(device))
        
        if image_batch is not None:
            with torch.no_grad():
                generated = model.generate(
                    scaffold_data=image_batch,
                    text_data=test_texts,
                    scaffold_modality='image',
                    output_modality='smiles',
                    num_beams=5,
                    temperature=0.8,
                    max_length=128
                )
            
            validity = calculate_validity(generated)
            uniqueness = calculate_uniqueness(generated)
            
            logger.info(f"生成的SMILES示例:")
            for i, gen_s in enumerate(generated[:3]):
                logger.info(f"  生成{i+1}: {gen_s}")
            
            logger.info(f"有效率: {validity:.2%}")
            logger.info(f"唯一性: {uniqueness:.2%}")
            
            results['image_to_smiles'] = {
                'validity': validity,
                'uniqueness': uniqueness,
                'examples': generated[:3]
            }
        else:
            logger.error("无法准备图像数据")
            results['image_to_smiles'] = {'error': "Image preparation failed"}
            
    except Exception as e:
        logger.error(f"IMAGE→SMILES失败: {e}")
        results['image_to_smiles'] = {'error': str(e)}
    
    # 测试3: GRAPH → SMILES
    logger.info("\n" + "="*60)
    logger.info("测试 GRAPH → SMILES")
    logger.info("="*60)
    
    try:
        # 准备图批次 - 作为列表
        graphs = []
        for smiles in test_smiles:
            graph = graph_processor.smiles_to_graph(smiles)
            if graph is None:
                from torch_geometric.data import Data
                # 创建默认图，注意特征维度应该是9
                graph = Data(
                    x=torch.randn(5, 9).to(device),
                    edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long).to(device),
                    edge_attr=torch.randn(4, 3).to(device)
                )
            graphs.append(graph)
        
        with torch.no_grad():
            generated = model.generate(
                scaffold_data=graphs,
                text_data=test_texts,
                scaffold_modality='graph',
                output_modality='smiles',
                num_beams=5,
                temperature=0.8,
                max_length=128
            )
        
        validity = calculate_validity(generated)
        uniqueness = calculate_uniqueness(generated)
        
        logger.info(f"生成的SMILES示例:")
        for i, gen_s in enumerate(generated[:3]):
            logger.info(f"  生成{i+1}: {gen_s}")
        
        logger.info(f"有效率: {validity:.2%}")
        logger.info(f"唯一性: {uniqueness:.2%}")
        
        results['graph_to_smiles'] = {
            'validity': validity,
            'uniqueness': uniqueness,
            'examples': generated[:3]
        }
        
    except Exception as e:
        logger.error(f"GRAPH→SMILES失败: {e}")
        results['graph_to_smiles'] = {'error': str(e)}
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"test_results/trained_model_test_{timestamp}.json"
    
    Path("test_results").mkdir(exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("测试完成")
    logger.info(f"结果已保存到: {output_file}")
    logger.info(f"{'='*60}")
    
    # 打印总结
    logger.info("\n📊 测试总结:")
    for test_name, result in results.items():
        if 'error' in result:
            logger.info(f"  {test_name}: ❌ 失败 - {result['error']}")
        else:
            logger.info(f"  {test_name}: ✅ 成功")
            logger.info(f"    - 有效率: {result['validity']:.2%}")
            logger.info(f"    - 唯一性: {result['uniqueness']:.2%}")

if __name__ == "__main__":
    main()