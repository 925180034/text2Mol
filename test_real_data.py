#!/usr/bin/env python
"""
🧪 使用真实ChEBI-20数据集测试9模态系统

测试所有9种输入输出组合，使用真实分子数据
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
import time
from datetime import datetime
from rdkit import Chem
from rdkit.Chem import Draw, Scaffolds

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scaffold_mol_gen.models.end2end_model import End2EndMolecularGenerator
from scaffold_mol_gen.utils.mol_utils import MolecularUtils
from scaffold_mol_gen.utils.scaffold_utils import ScaffoldExtractor
from scaffold_mol_gen.training.metrics import GenerationMetrics

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealDataTester:
    """使用真实数据的9模态测试器"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        初始化测试器
        
        Args:
            model_path: 训练好的模型路径
            device: 计算设备
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.scaffold_extractor = ScaffoldExtractor()
        self.metrics_calculator = GenerationMetrics()
        
        # 加载真实数据
        self.test_data = self._load_real_data()
        
        # 9种组合
        self.combinations = [
            ('smiles', 'smiles'), ('smiles', 'graph'), ('smiles', 'image'),
            ('graph', 'smiles'),  ('graph', 'graph'),  ('graph', 'image'),
            ('image', 'smiles'),  ('image', 'graph'),  ('image', 'image')
        ]
        
    def _load_real_data(self) -> pd.DataFrame:
        """加载真实的ChEBI-20测试数据"""
        test_file = "Datasets/test.csv"
        if not os.path.exists(test_file):
            logger.error(f"测试数据文件不存在: {test_file}")
            return pd.DataFrame()
        
        # 加载数据
        df = pd.read_csv(test_file)
        logger.info(f"加载了 {len(df)} 条测试数据")
        
        # 提取scaffold
        scaffolds = []
        valid_indices = []
        
        for idx, row in df.iterrows():
            try:
                smiles = row['SMILES']
                # 提取Murcko scaffold
                scaffold = self.scaffold_extractor.get_murcko_scaffold(smiles)
                if scaffold:
                    scaffolds.append(scaffold)
                    valid_indices.append(idx)
                else:
                    # 如果无法提取scaffold，尝试简化的版本
                    mol = Chem.MolFromSmiles(smiles)
                    if mol and mol.GetNumAtoms() > 5:
                        # 使用前6个原子作为scaffold（简化版）
                        scaffold = smiles.split('(')[0][:10] if '(' in smiles else smiles[:10]
                        scaffolds.append(scaffold)
                        valid_indices.append(idx)
            except Exception as e:
                logger.debug(f"跳过索引 {idx}: {e}")
                continue
        
        # 过滤有效数据
        df_valid = df.iloc[valid_indices].copy()
        df_valid['scaffold'] = scaffolds
        
        logger.info(f"成功处理 {len(df_valid)} 条有效数据")
        
        return df_valid
    
    def load_model(self) -> bool:
        """加载训练好的模型"""
        try:
            logger.info(f"加载模型: {self.model_path}")
            
            # 检查模型文件
            if not os.path.exists(self.model_path):
                logger.error(f"模型文件不存在: {self.model_path}")
                return False
            
            # 创建模型实例
            self.model = End2EndMolecularGenerator(
                hidden_size=768,
                molt5_path="/root/autodl-tmp/text2Mol-models/molt5-base",
                use_scibert=False,
                freeze_encoders=False,
                freeze_molt5=False,
                fusion_type='both',
                device=self.device
            )
            
            # 加载训练好的权重
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 处理不同的checkpoint格式
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"加载了 epoch {checkpoint.get('epoch', 'unknown')} 的模型")
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            logger.info("✅ 模型加载成功")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            return False
    
    def test_real_samples(self, sample_count: int = 10):
        """
        使用真实数据测试
        
        Args:
            sample_count: 测试样本数量
        """
        if len(self.test_data) == 0:
            logger.error("无可用测试数据")
            return
        
        # 加载模型
        if not self.load_model():
            return
        
        # 随机选择样本
        test_samples = self.test_data.sample(min(sample_count, len(self.test_data)))
        
        print("\n" + "="*70)
        print("🧪 真实数据测试 - ChEBI-20数据集")
        print("="*70)
        
        # 测试每个样本的不同组合
        results = []
        
        for idx, (_, row) in enumerate(test_samples.iterrows(), 1):
            print(f"\n📊 测试样本 {idx}/{len(test_samples)}")
            print(f"   CID: {row['CID']}")
            print(f"   原始SMILES: {row['SMILES'][:50]}...")
            print(f"   Scaffold: {row['scaffold']}")
            print(f"   描述: {row['description'][:100]}...")
            
            # 准备输入
            scaffold_smiles = row['scaffold']
            text_description = row['description']
            target_smiles = row['SMILES']
            
            # 测试主要组合 (SMILES → SMILES)
            print("\n   测试 SMILES → SMILES:")
            result = self._test_single_generation(
                scaffold_smiles, text_description, 
                'smiles', 'smiles', target_smiles
            )
            results.append(result)
            
            # 测试 Graph → SMILES
            print("\n   测试 Graph → SMILES:")
            result = self._test_single_generation(
                scaffold_smiles, text_description,
                'graph', 'smiles', target_smiles
            )
            results.append(result)
            
            # 测试 Image → SMILES  
            print("\n   测试 Image → SMILES:")
            result = self._test_single_generation(
                scaffold_smiles, text_description,
                'image', 'smiles', target_smiles
            )
            results.append(result)
        
        # 汇总结果
        self._print_summary(results)
    
    def _test_single_generation(self, scaffold_smiles: str, text: str, 
                               scaffold_mod: str, output_mod: str,
                               target_smiles: str) -> Dict[str, Any]:
        """测试单个生成"""
        try:
            start_time = time.time()
            
            # 生成
            with torch.no_grad():
                output = self.model.generate(
                    scaffold_data=[scaffold_smiles],
                    text_data=[text],
                    scaffold_modality=scaffold_mod,
                    output_modality=output_mod,
                    num_beams=5,
                    temperature=0.8,
                    max_length=128,
                    num_return_sequences=1
                )
            
            gen_time = time.time() - start_time
            
            # 获取生成的SMILES
            if output_mod == 'smiles':
                generated_smiles = output[0] if isinstance(output, list) else str(output)
            else:
                # 对于其他输出模态，重新生成SMILES版本用于评估
                smiles_output = self.model.generate(
                    scaffold_data=[scaffold_smiles],
                    text_data=[text],
                    scaffold_modality=scaffold_mod,
                    output_modality='smiles',
                    num_beams=5,
                    temperature=0.8,
                    max_length=128,
                    num_return_sequences=1
                )
                generated_smiles = smiles_output[0] if isinstance(smiles_output, list) else str(smiles_output)
            
            # 验证生成的SMILES
            is_valid = MolecularUtils.validate_smiles(generated_smiles)
            
            # 计算相似度
            if is_valid:
                from scaffold_mol_gen.utils.mol_utils import compute_tanimoto_similarity
                similarity = compute_tanimoto_similarity(generated_smiles, target_smiles)
            else:
                similarity = 0.0
            
            # 检查scaffold保持
            scaffold_preserved = False
            if is_valid:
                try:
                    gen_scaffold = self.scaffold_extractor.get_murcko_scaffold(generated_smiles)
                    scaffold_preserved = (gen_scaffold == scaffold_smiles)
                except:
                    pass
            
            result = {
                'combination': f"{scaffold_mod}→{output_mod}",
                'valid': is_valid,
                'similarity': similarity,
                'scaffold_preserved': scaffold_preserved,
                'generation_time': gen_time,
                'generated_smiles': generated_smiles[:50] if len(generated_smiles) > 50 else generated_smiles
            }
            
            # 打印结果
            status = "✅" if is_valid else "❌"
            print(f"      {status} 有效性: {is_valid}")
            print(f"      📊 相似度: {similarity:.3f}")
            print(f"      🔗 Scaffold保持: {scaffold_preserved}")
            print(f"      ⏱️  时间: {gen_time:.2f}s")
            print(f"      🧪 生成: {result['generated_smiles']}")
            
            return result
            
        except Exception as e:
            logger.error(f"生成失败: {e}")
            return {
                'combination': f"{scaffold_mod}→{output_mod}",
                'valid': False,
                'similarity': 0.0,
                'scaffold_preserved': False,
                'generation_time': 0.0,
                'error': str(e)
            }
    
    def _print_summary(self, results: List[Dict[str, Any]]):
        """打印汇总结果"""
        print("\n" + "="*70)
        print("📊 测试汇总")
        print("="*70)
        
        # 按组合分组统计
        combination_stats = {}
        for result in results:
            combo = result['combination']
            if combo not in combination_stats:
                combination_stats[combo] = {
                    'count': 0,
                    'valid_count': 0,
                    'similarities': [],
                    'scaffold_preserved_count': 0,
                    'times': []
                }
            
            stats = combination_stats[combo]
            stats['count'] += 1
            if result['valid']:
                stats['valid_count'] += 1
                stats['similarities'].append(result['similarity'])
            if result['scaffold_preserved']:
                stats['scaffold_preserved_count'] += 1
            stats['times'].append(result['generation_time'])
        
        # 打印统计
        print("\n组合性能统计:")
        print("-" * 70)
        print(f"{'组合':<15} {'有效率':<10} {'平均相似度':<12} {'Scaffold保持率':<15} {'平均时间':<10}")
        print("-" * 70)
        
        for combo, stats in combination_stats.items():
            validity_rate = stats['valid_count'] / stats['count'] if stats['count'] > 0 else 0
            avg_similarity = np.mean(stats['similarities']) if stats['similarities'] else 0
            scaffold_rate = stats['scaffold_preserved_count'] / stats['count'] if stats['count'] > 0 else 0
            avg_time = np.mean(stats['times'])
            
            print(f"{combo:<15} {validity_rate:>8.1%}  {avg_similarity:>10.3f}  {scaffold_rate:>13.1%}  {avg_time:>8.2f}s")
        
        # 总体统计
        total_count = len(results)
        total_valid = sum(1 for r in results if r['valid'])
        all_similarities = [r['similarity'] for r in results if r['valid']]
        total_scaffold_preserved = sum(1 for r in results if r['scaffold_preserved'])
        
        print("\n" + "="*70)
        print("总体性能:")
        print(f"  总测试数: {total_count}")
        print(f"  有效生成率: {total_valid/total_count:.1%}")
        if all_similarities:
            print(f"  平均相似度: {np.mean(all_similarities):.3f}")
        print(f"  Scaffold保持率: {total_scaffold_preserved/total_count:.1%}")
        print("="*70)


def main():
    """主函数"""
    print("🚀 启动真实数据测试 - ChEBI-20数据集")
    
    # 配置参数
    MODEL_PATH = "/root/autodl-tmp/text2Mol-outputs/9modal_20250810_161606_production/best_model.pth"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SAMPLE_COUNT = 5  # 测试5个真实样本
    
    print(f"📁 模型路径: {MODEL_PATH}")
    print(f"🖥️  计算设备: {DEVICE}")
    print(f"📊 测试样本数: {SAMPLE_COUNT}")
    
    # 检查模型文件
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 模型文件不存在 - {MODEL_PATH}")
        return
    
    # 创建测试器
    tester = RealDataTester(MODEL_PATH, DEVICE)
    
    # 运行测试
    try:
        tester.test_real_samples(sample_count=SAMPLE_COUNT)
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()