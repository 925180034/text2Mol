#!/usr/bin/env python3
"""
多模态分子生成系统全面评估
测试9种输入输出组合，计算10种评估指标
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append('/root/text2Mol/scaffold-mol-generation')

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# 导入模型和数据处理
from scaffold_mol_gen.models.end2end_model import End2EndMolecularGenerator

# 导入评估指标
from evaluation_metrics import (
    calculate_validity,
    calculate_uniqueness,
    calculate_novelty,
    calculate_bleu_score,
    calculate_exact_match,
    calculate_levenshtein_distance,
    calculate_fingerprint_similarity,
    calculate_fcd
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiModalEvaluator:
    """多模态分子生成评估器"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        self.model_path = model_path
        
        # 定义9种输入输出组合（实际上是7种，但我们测试所有可能的）
        self.io_combinations = [
            # 已实现的组合 (输出SMILES)
            ('smiles', 'smiles', '✅'),  # Scaffold(SMILES) + Text → SMILES
            ('graph', 'smiles', '✅'),   # Scaffold(Graph) + Text → SMILES
            ('image', 'smiles', '✅'),   # Scaffold(Image) + Text → SMILES
            
            # 待实现的组合 (需要额外解码器)
            ('smiles', 'graph', '🔄'),   # Scaffold(SMILES) + Text → Graph
            ('smiles', 'image', '🔄'),   # Scaffold(SMILES) + Text → Image
            ('graph', 'graph', '🔄'),    # Scaffold(Graph) + Text → Graph
            ('graph', 'image', '🔄'),    # Scaffold(Graph) + Text → Image
            ('image', 'graph', '🔄'),    # Scaffold(Image) + Text → Graph
            ('image', 'image', '🔄'),    # Scaffold(Image) + Text → Image
        ]
        
        # 10种评估指标
        self.metrics_list = [
            'validity',           # 1. 有效性
            'uniqueness',         # 2. 唯一性
            'novelty',           # 3. 新颖性
            'bleu_score',        # 4. BLEU分数
            'exact_match',       # 5. 精确匹配
            'levenshtein_dist',  # 6. 编辑距离
            'maccs_similarity',  # 7. MACCS指纹相似度
            'morgan_similarity', # 8. Morgan指纹相似度
            'rdkit_similarity',  # 9. RDKit指纹相似度
            'fcd_score'         # 10. FCD分数
        ]
        
        logger.info(f"初始化评估器 - 设备: {device}")
        logger.info(f"模型路径: {model_path}")
        
    def load_model(self):
        """加载训练好的模型"""
        logger.info("加载模型...")
        
        # 创建模型
        self.model = End2EndMolecularGenerator(
            hidden_size=768,
            molt5_path="/root/autodl-tmp/text2Mol-models/molt5-base",
            freeze_encoders=True,
            freeze_molt5=True,
            device=self.device
        )
        
        # 加载权重
        if Path(self.model_path).exists():
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"✅ 模型加载成功 - Epoch: {checkpoint.get('epoch', 'N/A')}")
        else:
            logger.warning(f"⚠️ 模型文件不存在: {self.model_path}")
            
        self.model.eval()
        self.model.to(self.device)
        
    def load_test_data(self, test_file: str = "Datasets/test.csv", sample_size: int = 100):
        """加载测试数据"""
        logger.info(f"加载测试数据: {test_file}")
        
        df = pd.read_csv(test_file)
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            logger.info(f"采样 {sample_size} 条数据进行测试")
            
        self.test_data = df
        logger.info(f"测试数据量: {len(self.test_data)}")
        
        return df
    
    def evaluate_io_combination(self, scaffold_modality: str, output_modality: str) -> Dict:
        """评估单个输入输出组合"""
        results = {
            'scaffold_modality': scaffold_modality,
            'output_modality': output_modality,
            'metrics': {},
            'examples': []
        }
        
        # 只评估输出为SMILES的组合（其他组合需要额外解码器）
        if output_modality != 'smiles':
            logger.info(f"⏭️ 跳过未实现的组合: {scaffold_modality} → {output_modality}")
            results['status'] = 'not_implemented'
            return results
            
        logger.info(f"评估组合: Scaffold({scaffold_modality}) + Text → {output_modality}")
        
        generated_smiles = []
        target_smiles = []
        
        # 批量生成
        batch_size = 8
        for i in tqdm(range(0, len(self.test_data), batch_size), desc=f"{scaffold_modality}→{output_modality}"):
            batch = self.test_data.iloc[i:i+batch_size]
            
            # 准备输入 - 使用正确的列名
            # test.csv 使用 'SMILES' 作为scaffold，'description' 作为文本
            smiles_list = batch['SMILES'].tolist()
            text_list = batch['description'].tolist()
            target_list = batch['SMILES'].tolist()  # 目标也是SMILES
            
            # 根据scaffold模态转换输入数据
            if scaffold_modality == 'smiles':
                scaffold_list = smiles_list
            elif scaffold_modality == 'graph':
                # 将SMILES转换为图
                from scaffold_mol_gen.data.multimodal_preprocessor import MultiModalPreprocessor
                preprocessor = MultiModalPreprocessor()
                scaffold_list = []
                for smiles in smiles_list:
                    graph = preprocessor.smiles_to_graph(smiles)
                    if graph is not None:
                        scaffold_list.append(graph)
                    else:
                        # 如果转换失败，跳过这个样本
                        logger.warning(f"无法将SMILES转换为图: {smiles}")
                        continue
                if not scaffold_list:
                    logger.error("所有SMILES到图的转换都失败了")
                    continue
            elif scaffold_modality == 'image':
                # 将SMILES转换为图像
                from scaffold_mol_gen.data.multimodal_preprocessor import MultiModalPreprocessor
                preprocessor = MultiModalPreprocessor()
                scaffold_list = []
                for smiles in smiles_list:
                    image = preprocessor.smiles_to_image(smiles)
                    if image is not None:
                        # 转换为tensor
                        import torchvision.transforms as transforms
                        transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225])
                        ])
                        if isinstance(image, np.ndarray):
                            image_tensor = transform(image)
                            scaffold_list.append(image_tensor)
                    else:
                        logger.warning(f"无法将SMILES转换为图像: {smiles}")
                        continue
                if not scaffold_list:
                    logger.error("所有SMILES到图像的转换都失败了")
                    continue
                # 将图像列表转换为批次tensor
                scaffold_list = torch.stack(scaffold_list).to(self.device)
            else:
                raise ValueError(f"不支持的scaffold模态: {scaffold_modality}")
            
            try:
                with torch.no_grad():
                    # 生成SMILES
                    output = self.model.generate(
                        scaffold_data=scaffold_list,
                        text_data=text_list,
                        scaffold_modality=scaffold_modality,
                        output_modality=output_modality,
                        num_beams=5,
                        temperature=0.8,
                        max_length=128
                    )
                    
                    if isinstance(output, list):
                        generated_smiles.extend(output)
                    else:
                        generated_smiles.extend(output.tolist() if hasattr(output, 'tolist') else [output])
                        
                    target_smiles.extend(target_list)
                    
                    # 保存前5个例子
                    if len(results['examples']) < 5:
                        for j in range(min(len(scaffold_list), 5 - len(results['examples']))):
                            results['examples'].append({
                                'scaffold': scaffold_list[j],
                                'text': text_list[j][:50] + '...' if len(text_list[j]) > 50 else text_list[j],
                                'target': target_list[j],
                                'generated': output[j] if isinstance(output, list) else str(output)
                            })
                            
            except Exception as e:
                logger.error(f"生成错误: {e}")
                continue
        
        # 计算所有指标
        if generated_smiles and target_smiles:
            results['metrics'] = self.calculate_all_metrics(generated_smiles, target_smiles)
            results['status'] = 'success'
        else:
            results['status'] = 'failed'
            
        return results
    
    def calculate_all_metrics(self, generated: List[str], target: List[str]) -> Dict:
        """计算所有10种评估指标"""
        metrics = {}
        
        try:
            # 1. 有效性
            metrics['validity'] = calculate_validity(generated)
            
            # 2. 唯一性
            metrics['uniqueness'] = calculate_uniqueness(generated)
            
            # 3. 新颖性
            metrics['novelty'] = calculate_novelty(generated, target)
            
            # 4. BLEU分数
            metrics['bleu_score'] = calculate_bleu_score(generated, target)
            
            # 5. 精确匹配
            metrics['exact_match'] = calculate_exact_match(generated, target)
            
            # 6. 编辑距离
            metrics['levenshtein_dist'] = calculate_levenshtein_distance(generated, target)
            
            # 7-9. 指纹相似度
            metrics['maccs_similarity'] = calculate_fingerprint_similarity(
                generated, target, fingerprint_type='maccs'
            )
            metrics['morgan_similarity'] = calculate_fingerprint_similarity(
                generated, target, fingerprint_type='morgan'
            )
            metrics['rdkit_similarity'] = calculate_fingerprint_similarity(
                generated, target, fingerprint_type='rdkit'
            )
            
            # 10. FCD分数 (需要预训练的ChemNet模型，可能较慢)
            try:
                metrics['fcd_score'] = calculate_fcd(generated, target)
            except:
                metrics['fcd_score'] = -1  # FCD计算失败时返回-1
                
        except Exception as e:
            logger.error(f"指标计算错误: {e}")
            
        return metrics
    
    def run_comprehensive_evaluation(self):
        """运行全面评估"""
        logger.info("=" * 80)
        logger.info("🔬 多模态分子生成系统全面评估")
        logger.info("=" * 80)
        
        # 加载模型
        self.load_model()
        
        # 加载测试数据
        self.load_test_data(sample_size=100)  # 使用100个样本进行快速评估
        
        # 存储所有结果
        all_results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': self.model_path,
            'test_samples': len(self.test_data),
            'combinations': []
        }
        
        # 评估每个输入输出组合
        for scaffold_mod, output_mod, status in self.io_combinations:
            logger.info(f"\n{'='*60}")
            logger.info(f"测试组合 {status}: Scaffold({scaffold_mod}) + Text → {output_mod}")
            logger.info(f"{'='*60}")
            
            result = self.evaluate_io_combination(scaffold_mod, output_mod)
            all_results['combinations'].append(result)
            
            # 打印结果
            if result['status'] == 'success':
                self.print_metrics(result['metrics'])
                if result['examples']:
                    self.print_examples(result['examples'])
        
        # 生成总结报告
        self.generate_summary_report(all_results)
        
        # 保存详细结果
        output_file = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"\n📊 详细结果已保存: {output_file}")
        
        return all_results
    
    def print_metrics(self, metrics: Dict):
        """打印评估指标"""
        logger.info("\n📈 评估指标:")
        logger.info("-" * 40)
        
        for i, (key, value) in enumerate(metrics.items(), 1):
            if isinstance(value, float):
                logger.info(f"{i:2}. {key:20s}: {value:.4f}")
            else:
                logger.info(f"{i:2}. {key:20s}: {value}")
    
    def print_examples(self, examples: List[Dict]):
        """打印生成示例"""
        logger.info("\n🔍 生成示例:")
        logger.info("-" * 40)
        
        for i, ex in enumerate(examples[:3], 1):
            logger.info(f"\n示例 {i}:")
            logger.info(f"  Scaffold: {ex['scaffold'][:50]}...")
            logger.info(f"  文本: {ex['text']}")
            logger.info(f"  目标: {ex['target'][:50]}...")
            logger.info(f"  生成: {ex['generated'][:50]}...")
            
            # 验证生成的SMILES
            mol = Chem.MolFromSmiles(ex['generated'])
            if mol:
                logger.info(f"  ✅ 有效SMILES")
            else:
                logger.info(f"  ❌ 无效SMILES")
    
    def generate_summary_report(self, all_results: Dict):
        """生成总结报告"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 评估总结报告")
        logger.info("=" * 80)
        
        # 统计实现的组合
        implemented = [r for r in all_results['combinations'] if r['status'] == 'success']
        
        if implemented:
            logger.info(f"\n✅ 成功评估的组合: {len(implemented)}/9")
            
            # 计算平均指标
            avg_metrics = {}
            for metric_name in self.metrics_list:
                values = [r['metrics'].get(metric_name, 0) for r in implemented if metric_name in r['metrics']]
                if values:
                    avg_metrics[metric_name] = np.mean(values)
            
            logger.info("\n📊 平均性能指标:")
            logger.info("-" * 40)
            
            # 重点指标
            key_metrics = ['validity', 'uniqueness', 'novelty', 'exact_match', 'morgan_similarity']
            for metric in key_metrics:
                if metric in avg_metrics:
                    value = avg_metrics[metric]
                    # 根据指标类型显示不同的表情
                    if metric == 'validity':
                        emoji = "🎯" if value > 0.6 else "⚠️"
                    elif metric == 'uniqueness':
                        emoji = "💎" if value > 0.8 else "📊"
                    elif metric == 'novelty':
                        emoji = "🌟" if value > 0.5 else "📈"
                    elif metric == 'exact_match':
                        emoji = "✅" if value > 0.3 else "📊"
                    else:
                        emoji = "📊"
                    
                    logger.info(f"{emoji} {metric:20s}: {value:.4f} ({value*100:.2f}%)")
            
            # 其他指标
            logger.info("\n📈 其他指标:")
            for metric, value in avg_metrics.items():
                if metric not in key_metrics:
                    logger.info(f"   {metric:20s}: {value:.4f}")
        
        # 总结
        logger.info("\n" + "=" * 80)
        logger.info("🎯 关键发现:")
        
        if implemented:
            validity_scores = [r['metrics'].get('validity', 0) for r in implemented]
            avg_validity = np.mean(validity_scores) if validity_scores else 0
            
            if avg_validity > 0.6:
                logger.info(f"✅ SMILES有效性达到 {avg_validity*100:.1f}% - 显著改进！")
            else:
                logger.info(f"⚠️ SMILES有效性为 {avg_validity*100:.1f}% - 需要进一步优化")
                
            # 最佳组合
            best_combo = max(implemented, key=lambda x: x['metrics'].get('validity', 0))
            logger.info(f"🏆 最佳组合: {best_combo['scaffold_modality']} → {best_combo['output_modality']}")
            logger.info(f"   有效性: {best_combo['metrics'].get('validity', 0)*100:.1f}%")
        
        logger.info("=" * 80)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='多模态分子生成系统全面评估')
    parser.add_argument('--model-path', type=str,
                       default='/root/autodl-tmp/text2Mol-outputs/optimized_20250809_105726/best_model.pt',
                       help='模型路径')
    parser.add_argument('--test-file', type=str,
                       default='Datasets/test.csv',
                       help='测试数据文件')
    parser.add_argument('--sample-size', type=int, default=100,
                       help='测试样本数量')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = MultiModalEvaluator(
        model_path=args.model_path,
        device=args.device
    )
    
    # 运行评估
    results = evaluator.run_comprehensive_evaluation()
    
    logger.info("\n✅ 评估完成！")
    

if __name__ == "__main__":
    main()