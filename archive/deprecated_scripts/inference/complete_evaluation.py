#!/usr/bin/env python3
"""
完整的模型评估脚本 - 计算所有9个指标并生成实验报告
"""

import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw

from simple_inference import SimpleInference
from scaffold_mol_gen.training.metrics import MolecularMetrics

class CompleteEvaluator:
    def __init__(self, model_path, experiment_name=None):
        """初始化完整评估器"""
        self.model_path = model_path
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 创建实验目录结构
        self.exp_dir = Path(f"experiments/{self.experiment_name}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 子目录
        (self.exp_dir / "visualizations").mkdir(exist_ok=True)
        (self.exp_dir / "molecular_images").mkdir(exist_ok=True)
        (self.exp_dir / "metrics").mkdir(exist_ok=True)
        (self.exp_dir / "logs").mkdir(exist_ok=True)
        
        # 初始化推理引擎和评估器
        self.inference = SimpleInference(model_path)
        self.metrics = MolecularMetrics()
        
        print(f"实验目录创建成功: {self.exp_dir}")
    
    def evaluate_test_set(self, test_file, num_samples=None, save_examples=True):
        """在测试集上进行完整评估"""
        
        # 加载测试数据
        print(f"\n加载测试数据: {test_file}")
        test_df = pd.read_csv(test_file)
        
        if num_samples:
            test_df = test_df.head(num_samples)
            print(f"评估前 {num_samples} 个样本")
        else:
            print(f"评估全部 {len(test_df)} 个样本")
        
        # 存储结果
        results = []
        all_predictions = []
        all_targets = []
        all_texts = []
        all_scaffolds = []
        
        # 推理进度条
        print("\n开始模型推理...")
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="生成分子"):
            try:
                # 生成分子
                generated_smiles = self.inference.generate_molecule(
                    scaffold_smiles=row['scaffold'],
                    text_description=row['text'],
                    num_beams=5,
                    temperature=1.0
                )
                
                # 记录结果
                result = {
                    'idx': idx,
                    'scaffold': row['scaffold'],
                    'text': row['text'],
                    'target': row['SMILES'],
                    'generated': generated_smiles,
                    'success': True
                }
                
            except Exception as e:
                result = {
                    'idx': idx,
                    'scaffold': row['scaffold'],
                    'text': row['text'],
                    'target': row['SMILES'],
                    'generated': '',
                    'success': False,
                    'error': str(e)
                }
                generated_smiles = ''
            
            results.append(result)
            all_predictions.append(generated_smiles)
            all_targets.append(row['SMILES'])
            all_texts.append(row['text'])
            all_scaffolds.append(row['scaffold'])
        
        # 保存推理结果
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.exp_dir / 'inference_results.csv', index=False)
        
        # 计算所有9个指标
        print("\n计算所有评估指标...")
        evaluation_results = self._calculate_all_metrics(
            all_predictions, all_targets, all_scaffolds
        )
        
        # 保存评估结果
        self._save_evaluation_results(evaluation_results, results_df)
        
        # 生成可视化
        if save_examples:
            self._generate_visualizations(results_df, evaluation_results)
        
        # 生成实验报告
        self._generate_report(evaluation_results, results_df)
        
        return results_df, evaluation_results
    
    def _calculate_all_metrics(self, predictions, targets, scaffolds):
        """计算所有9个评估指标"""
        metrics_dict = {}
        
        print("\n计算基础指标...")
        # 1. Validity (有效性)
        metrics_dict['validity'] = self.metrics.validity(predictions)
        print(f"✓ Validity: {metrics_dict['validity']:.4f}")
        
        # 2. Uniqueness (唯一性)
        valid_preds = [p for p in predictions if p and Chem.MolFromSmiles(p)]
        metrics_dict['uniqueness'] = self.metrics.uniqueness(valid_preds)
        print(f"✓ Uniqueness: {metrics_dict['uniqueness']:.4f}")
        
        # 3. Novelty (新颖性)
        metrics_dict['novelty'] = self.metrics.novelty(valid_preds, scaffolds)
        print(f"✓ Novelty: {metrics_dict['novelty']:.4f}")
        
        print("\n计算文本相似度指标...")
        # 4. BLEU Score
        metrics_dict['bleu'] = self.metrics.bleu_score(predictions, targets)
        print(f"✓ BLEU Score: {metrics_dict['bleu']:.4f}")
        
        # 5. Exact Match (精确匹配)
        metrics_dict['exact_match'] = self.metrics.exact_match_score(predictions, targets)
        print(f"✓ Exact Match: {metrics_dict['exact_match']:.4f}")
        
        # 6. Levenshtein Distance
        metrics_dict['levenshtein'] = self.metrics.levenshtein_distance(predictions, targets)
        print(f"✓ Levenshtein Distance: {metrics_dict['levenshtein']:.4f}")
        
        print("\n计算分子相似度指标...")
        # 7. MACCS Fingerprint Similarity
        try:
            metrics_dict['maccs_similarity'] = self.metrics.maccs_similarity(predictions, targets)
            print(f"✓ MACCS Similarity: {metrics_dict['maccs_similarity']:.4f}")
        except Exception as e:
            metrics_dict['maccs_similarity'] = None
            print(f"✗ MACCS Similarity 计算失败: {e}")
        
        # 8. Morgan Fingerprint Similarity
        try:
            metrics_dict['morgan_similarity'] = self.metrics.morgan_similarity(predictions, targets)
            print(f"✓ Morgan Similarity: {metrics_dict['morgan_similarity']:.4f}")
        except Exception as e:
            metrics_dict['morgan_similarity'] = None
            print(f"✗ Morgan Similarity 计算失败: {e}")
        
        # 9. RDK Fingerprint Similarity
        try:
            metrics_dict['rdk_similarity'] = self.metrics.rdk_similarity(predictions, targets)
            print(f"✓ RDK Similarity: {metrics_dict['rdk_similarity']:.4f}")
        except Exception as e:
            metrics_dict['rdk_similarity'] = None
            print(f"✗ RDK Similarity 计算失败: {e}")
        
        # 10. FCD (如果可用)
        try:
            print("\n计算FCD指标...")
            metrics_dict['fcd'] = self.metrics.calculate_fcd(valid_preds, targets)
            print(f"✓ FCD: {metrics_dict['fcd']:.4f}")
        except Exception as e:
            metrics_dict['fcd'] = None
            print(f"✗ FCD 计算失败（可能需要安装额外依赖）: {e}")
        
        # 额外统计信息
        metrics_dict['total_samples'] = len(predictions)
        metrics_dict['valid_samples'] = len(valid_preds)
        metrics_dict['success_rate'] = len(valid_preds) / len(predictions) if predictions else 0
        
        return metrics_dict
    
    def _save_evaluation_results(self, evaluation_results, results_df):
        """保存评估结果"""
        # 保存JSON格式
        with open(self.exp_dir / 'metrics' / 'all_metrics.json', 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # 保存为易读的文本格式
        with open(self.exp_dir / 'metrics' / 'metrics_summary.txt', 'w') as f:
            f.write("="*60 + "\n")
            f.write("分子生成模型评估结果\n")
            f.write(f"实验名称: {self.experiment_name}\n")
            f.write(f"模型路径: {self.model_path}\n")
            f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            
            f.write("【基础指标】\n")
            f.write(f"1. Validity (有效性): {evaluation_results.get('validity', 'N/A'):.4f}\n")
            f.write(f"2. Uniqueness (唯一性): {evaluation_results.get('uniqueness', 'N/A'):.4f}\n")
            f.write(f"3. Novelty (新颖性): {evaluation_results.get('novelty', 'N/A'):.4f}\n\n")
            
            f.write("【文本相似度指标】\n")
            f.write(f"4. BLEU Score: {evaluation_results.get('bleu', 'N/A'):.4f}\n")
            f.write(f"5. Exact Match (精确匹配): {evaluation_results.get('exact_match', 'N/A'):.4f}\n")
            f.write(f"6. Levenshtein Distance: {evaluation_results.get('levenshtein', 'N/A'):.4f}\n\n")
            
            f.write("【分子相似度指标】\n")
            f.write(f"7. MACCS Similarity: {evaluation_results.get('maccs_similarity', 'N/A'):.4f}\n")
            f.write(f"8. Morgan Similarity: {evaluation_results.get('morgan_similarity', 'N/A'):.4f}\n")
            f.write(f"9. RDK Similarity: {evaluation_results.get('rdk_similarity', 'N/A'):.4f}\n")
            
            if evaluation_results.get('fcd') is not None:
                f.write(f"10. FCD Score: {evaluation_results.get('fcd', 'N/A'):.4f}\n\n")
            
            f.write("【统计信息】\n")
            f.write(f"总样本数: {evaluation_results.get('total_samples', 0)}\n")
            f.write(f"有效生成数: {evaluation_results.get('valid_samples', 0)}\n")
            f.write(f"成功率: {evaluation_results.get('success_rate', 0):.2%}\n")
    
    def _generate_visualizations(self, results_df, evaluation_results):
        """生成可视化图表"""
        print("\n生成可视化图表...")
        
        # 1. 指标柱状图
        plt.figure(figsize=(12, 6))
        metrics_names = ['Validity', 'Uniqueness', 'Novelty', 'BLEU', 'Exact Match', 
                        'MACCS Sim', 'Morgan Sim', 'RDK Sim']
        metrics_values = [
            evaluation_results.get('validity', 0),
            evaluation_results.get('uniqueness', 0),
            evaluation_results.get('novelty', 0),
            evaluation_results.get('bleu', 0),
            evaluation_results.get('exact_match', 0),
            evaluation_results.get('maccs_similarity', 0),
            evaluation_results.get('morgan_similarity', 0),
            evaluation_results.get('rdk_similarity', 0)
        ]
        
        bars = plt.bar(metrics_names, metrics_values)
        plt.title('分子生成模型评估指标', fontsize=16)
        plt.ylabel('分数', fontsize=12)
        plt.ylim(0, 1.1)
        
        # 添加数值标签
        for bar, value in zip(bars, metrics_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.exp_dir / 'visualizations' / 'metrics_bar_chart.png', dpi=300)
        plt.close()
        
        # 2. 成功示例可视化
        success_samples = results_df[results_df['success'] == True].head(10)
        for idx, row in success_samples.iterrows():
            try:
                # 创建分子可视化
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # 绘制骨架
                scaffold_mol = Chem.MolFromSmiles(row['scaffold'])
                if scaffold_mol:
                    img1 = Draw.MolToImage(scaffold_mol)
                    ax1.imshow(img1)
                    ax1.set_title("输入骨架")
                    ax1.axis('off')
                
                # 绘制生成的分子
                generated_mol = Chem.MolFromSmiles(row['generated'])
                if generated_mol:
                    img2 = Draw.MolToImage(generated_mol)
                    ax2.imshow(img2)
                    ax2.set_title("生成的完整分子")
                    ax2.axis('off')
                
                plt.tight_layout()
                plt.savefig(self.exp_dir / 'molecular_images' / f'example_{idx}.png')
                plt.close(fig)
            except:
                pass
        
        print(f"✓ 可视化图表已保存到: {self.exp_dir / 'visualizations'}")
    
    def _generate_report(self, evaluation_results, results_df):
        """生成完整的实验报告"""
        report_path = self.exp_dir / 'experiment_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# 分子生成模型实验报告\n\n")
            f.write(f"**实验名称**: {self.experiment_name}\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**模型路径**: `{self.model_path}`\n\n")
            
            f.write("## 1. 评估指标汇总\n\n")
            f.write("| 指标类别 | 指标名称 | 分数 | 说明 |\n")
            f.write("|---------|---------|------|------|\n")
            f.write(f"| 基础指标 | Validity | {evaluation_results.get('validity', 0):.4f} | 生成分子的有效性 |\n")
            f.write(f"| 基础指标 | Uniqueness | {evaluation_results.get('uniqueness', 0):.4f} | 生成分子的唯一性 |\n")
            f.write(f"| 基础指标 | Novelty | {evaluation_results.get('novelty', 0):.4f} | 生成分子的新颖性 |\n")
            f.write(f"| 文本相似度 | BLEU Score | {evaluation_results.get('bleu', 0):.4f} | 生成序列与目标的相似度 |\n")
            f.write(f"| 文本相似度 | Exact Match | {evaluation_results.get('exact_match', 0):.4f} | 完全匹配的比例 |\n")
            f.write(f"| 文本相似度 | Levenshtein | {evaluation_results.get('levenshtein', 0):.4f} | 编辑距离（归一化） |\n")
            f.write(f"| 分子相似度 | MACCS | {evaluation_results.get('maccs_similarity', 0):.4f} | MACCS指纹相似度 |\n")
            f.write(f"| 分子相似度 | Morgan | {evaluation_results.get('morgan_similarity', 0):.4f} | Morgan指纹相似度 |\n")
            f.write(f"| 分子相似度 | RDK | {evaluation_results.get('rdk_similarity', 0):.4f} | RDKit指纹相似度 |\n")
            
            if evaluation_results.get('fcd') is not None:
                f.write(f"| 分布距离 | FCD | {evaluation_results.get('fcd', 0):.4f} | Fréchet ChemNet Distance |\n")
            
            f.write("\n## 2. 生成示例\n\n")
            success_samples = results_df[results_df['success'] == True].head(5)
            for i, (_, row) in enumerate(success_samples.iterrows(), 1):
                f.write(f"### 示例 {i}\n")
                f.write(f"- **骨架**: `{row['scaffold']}`\n")
                f.write(f"- **描述**: {row['text'][:100]}...\n")
                f.write(f"- **目标分子**: `{row['target']}`\n")
                f.write(f"- **生成分子**: `{row['generated']}`\n")
                f.write(f"- **可视化**: ![](molecular_images/example_{row['idx']}.png)\n\n")
            
            f.write("## 3. 统计信息\n\n")
            f.write(f"- 总测试样本数: {evaluation_results.get('total_samples', 0)}\n")
            f.write(f"- 成功生成数: {evaluation_results.get('valid_samples', 0)}\n")
            f.write(f"- 生成成功率: {evaluation_results.get('success_rate', 0):.2%}\n")
            
            f.write("\n## 4. 文件列表\n\n")
            f.write("- `inference_results.csv` - 详细的推理结果\n")
            f.write("- `metrics/all_metrics.json` - 所有指标的JSON格式\n")
            f.write("- `metrics/metrics_summary.txt` - 指标摘要\n")
            f.write("- `visualizations/metrics_bar_chart.png` - 指标可视化\n")
            f.write("- `molecular_images/` - 生成分子的可视化图像\n")
        
        print(f"\n✓ 实验报告已生成: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='完整的分子生成模型评估')
    parser.add_argument('--model', type=str, default='/root/autodl-tmp/safe_fast_checkpoints/best_model.pt',
                       help='模型路径')
    parser.add_argument('--test', type=str, default='Datasets/test.csv',
                       help='测试数据路径')
    parser.add_argument('--name', type=str, default=None,
                       help='实验名称（默认使用时间戳）')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='测试样本数量（默认全部）')
    parser.add_argument('--save_examples', action='store_true', default=True,
                       help='是否保存示例可视化')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = CompleteEvaluator(
        model_path=args.model,
        experiment_name=args.name
    )
    
    # 运行完整评估
    results_df, evaluation_results = evaluator.evaluate_test_set(
        test_file=args.test,
        num_samples=args.num_samples,
        save_examples=args.save_examples
    )
    
    print("\n" + "="*60)
    print("评估完成！")
    print(f"所有结果已保存到: {evaluator.exp_dir}")
    print("="*60)

if __name__ == '__main__':
    main()