#!/usr/bin/env python3
"""
批量推理脚本 - 在测试集上运行推理并计算评估指标
"""

import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

from inference import MolecularInference
from scaffold_mol_gen.training.metrics import MolecularMetrics

def run_batch_inference(model_path, test_file, output_dir, num_samples=None):
    """在测试集上批量运行推理"""
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载测试数据
    print(f"加载测试数据: {test_file}")
    test_df = pd.read_csv(test_file)
    
    if num_samples:
        test_df = test_df.head(num_samples)
        print(f"只测试前 {num_samples} 个样本")
    
    # 初始化推理引擎和评估器
    inference = MolecularInference(model_path)
    metrics = MolecularMetrics()
    
    # 存储结果
    results = []
    all_predictions = []
    all_targets = []
    
    # 批量推理
    print(f"\n开始推理 {len(test_df)} 个样本...")
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        try:
            # 生成分子
            generated_smiles = inference.generate_molecule(
                scaffold=row['scaffold'],
                text_description=row['text'],
                scaffold_type='smiles',
                num_beams=5,
                temperature=1.0
            )
            
            # 记录结果
            result = {
                'idx': idx,
                'scaffold': row['scaffold'],
                'text': row['text'],
                'target': row['SMILES'],
                'generated': generated_smiles
            }
            results.append(result)
            
            all_predictions.append(generated_smiles)
            all_targets.append(row['SMILES'])
            
        except Exception as e:
            print(f"\n样本 {idx} 生成失败: {str(e)}")
            result = {
                'idx': idx,
                'scaffold': row['scaffold'],
                'text': row['text'],
                'target': row['SMILES'],
                'generated': '',
                'error': str(e)
            }
            results.append(result)
            all_predictions.append('')
            all_targets.append(row['SMILES'])
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'inference_results.csv', index=False)
    print(f"\n结果已保存到: {output_dir / 'inference_results.csv'}")
    
    # 计算评估指标
    print("\n计算评估指标...")
    evaluation_results = {}
    
    # 基础指标
    valid_preds = [p for p in all_predictions if p]
    evaluation_results['validity'] = metrics.validity(all_predictions)
    evaluation_results['uniqueness'] = metrics.uniqueness(valid_preds)
    evaluation_results['novelty'] = metrics.novelty(valid_preds, all_targets)
    
    # 文本相似度指标
    evaluation_results['exact_match'] = metrics.exact_match_score(all_predictions, all_targets)
    evaluation_results['bleu'] = metrics.bleu_score(all_predictions, all_targets)
    evaluation_results['levenshtein'] = metrics.levenshtein_distance(all_predictions, all_targets)
    
    # 分子相似度指标
    try:
        evaluation_results['maccs_similarity'] = metrics.maccs_similarity(all_predictions, all_targets)
        evaluation_results['morgan_similarity'] = metrics.morgan_similarity(all_predictions, all_targets)
        evaluation_results['rdk_similarity'] = metrics.rdk_similarity(all_predictions, all_targets)
    except Exception as e:
        print(f"分子相似度计算失败: {e}")
    
    # 打印评估结果
    print("\n=== 评估结果 ===")
    for metric, value in evaluation_results.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    # 保存评估结果
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    print(f"\n评估结果已保存到: {output_dir / 'evaluation_results.json'}")
    
    # 生成示例
    print("\n=== 生成示例 ===")
    for i in range(min(5, len(results_df))):
        row = results_df.iloc[i]
        print(f"\n示例 {i+1}:")
        print(f"骨架: {row['scaffold']}")
        print(f"描述: {row['text'][:100]}...")
        print(f"目标: {row['target']}")
        print(f"生成: {row['generated']}")
        if 'error' in row and pd.notna(row.get('error')):
            print(f"错误: {row['error']}")
    
    return results_df, evaluation_results

def main():
    parser = argparse.ArgumentParser(description='批量分子生成推理')
    parser.add_argument('--model', type=str, default='/root/autodl-tmp/safe_fast_checkpoints/best_model.pt',
                       help='模型路径')
    parser.add_argument('--test', type=str, default='Datasets/test.csv',
                       help='测试数据路径')
    parser.add_argument('--output', type=str, default='inference_output',
                       help='输出目录')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='测试样本数量（默认全部）')
    
    args = parser.parse_args()
    
    # 运行批量推理
    results_df, evaluation_results = run_batch_inference(
        model_path=args.model,
        test_file=args.test,
        output_dir=args.output,
        num_samples=args.num_samples
    )
    
    print("\n推理完成！")

if __name__ == '__main__':
    main()