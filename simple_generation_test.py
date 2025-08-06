#!/usr/bin/env python3
"""
简单的生成测试，验证模型是否真实生成新分子
"""

import torch
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from rdkit import Chem

# 导入模型
from scaffold_mol_gen.models.end2end_model import End2EndMolecularGenerator

def test_single_generation():
    """测试单个样本的生成过程"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 1. 加载模型
    print("加载模型...")
    molt5_path = '/root/autodl-tmp/text2Mol-models/MolT5-Large-Caption2SMILES'
    model_path = '/root/autodl-tmp/text2Mol-outputs/fast_training/smiles/final_model.pt'
    
    model = End2EndMolecularGenerator(
        molt5_path=molt5_path,
        fusion_type='both',
        device=str(device)
    )
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("✅ 模型加载完成")
    
    # 2. 准备测试数据
    print("\n准备测试数据...")
    
    # 加载一个样本
    csv_path = Path('/root/text2Mol/scaffold-mol-generation/Datasets/test.csv')
    df = pd.read_csv(csv_path)
    
    # 选择第一个样本
    sample = df.iloc[0]
    scaffold_smiles = sample['SMILES']
    text_description = sample['description']
    target_smiles = sample['SMILES']
    
    print(f"Scaffold SMILES: {scaffold_smiles}")
    print(f"Text: {text_description[:100]}...")
    print(f"Target SMILES: {target_smiles}")
    
    # 3. 多次生成测试
    print("\n🔬 开始多次生成测试...")
    
    generated_results = []
    
    with torch.no_grad():
        for i in range(5):
            print(f"\n--- 生成 #{i+1} ---")
            
            try:
                # 生成分子
                output = model.generate(
                    scaffold_data=scaffold_smiles,
                    text_data=text_description,
                    scaffold_modality='smiles',
                    output_modality='smiles',
                    num_beams=3,
                    temperature=0.8,  # 稍高的温度增加随机性
                    max_length=128
                )
                
                if isinstance(output, list) and len(output) > 0:
                    generated_smiles = output[0]
                elif isinstance(output, str):
                    generated_smiles = output
                else:
                    generated_smiles = "GENERATION_FAILED"
                
                print(f"Generated: {generated_smiles}")
                print(f"Target:    {target_smiles}")
                print(f"Match:     {generated_smiles == target_smiles}")
                
                # 检查分子有效性
                if generated_smiles != "GENERATION_FAILED":
                    mol = Chem.MolFromSmiles(generated_smiles)
                    valid = mol is not None
                    print(f"Valid:     {valid}")
                    
                    if valid and generated_smiles != target_smiles:
                        print(f"✅ 生成了不同的有效分子！")
                else:
                    print("❌ 生成失败")
                
                generated_results.append({
                    'run': i+1,
                    'generated': generated_smiles,
                    'target': target_smiles,
                    'exact_match': generated_smiles == target_smiles,
                    'valid': generated_smiles != "GENERATION_FAILED"
                })
                
            except Exception as e:
                print(f"❌ 生成出错: {e}")
                import traceback
                traceback.print_exc()
                
                generated_results.append({
                    'run': i+1,
                    'generated': "ERROR",
                    'target': target_smiles,
                    'exact_match': False,
                    'valid': False,
                    'error': str(e)
                })
    
    # 4. 分析结果
    print("\n" + "="*70)
    print("📊 生成结果分析")
    print("="*70)
    
    total_runs = len(generated_results)
    successful_generations = sum(1 for r in generated_results if r['valid'])
    exact_matches = sum(1 for r in generated_results if r['exact_match'])
    unique_generations = len(set(r['generated'] for r in generated_results if r['valid']))
    
    print(f"总运行次数: {total_runs}")
    print(f"成功生成: {successful_generations}")
    print(f"精确匹配: {exact_matches}")
    print(f"唯一生成数: {unique_generations}")
    print(f"成功率: {successful_generations/total_runs:.1%}")
    print(f"匹配率: {exact_matches/total_runs:.1%}")
    
    if exact_matches == total_runs:
        print("\n⚠️ 所有生成都与目标完全匹配 - 可能存在fallback问题")
    elif exact_matches == 0:
        print("\n✅ 没有精确匹配 - 模型正在生成新分子")
    else:
        print(f"\n🔍 部分匹配 ({exact_matches}/{total_runs}) - 需要进一步调查")
    
    # 5. 保存详细结果
    import json
    results_file = '/root/text2Mol/scaffold-mol-generation/simple_generation_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'test_sample': {
                'scaffold': scaffold_smiles,
                'text': text_description,
                'target': target_smiles
            },
            'generation_results': generated_results,
            'summary': {
                'total_runs': total_runs,
                'successful_generations': successful_generations,
                'exact_matches': exact_matches,
                'unique_generations': unique_generations,
                'success_rate': successful_generations/total_runs,
                'match_rate': exact_matches/total_runs
            }
        }, indent=2)
    
    print(f"\n💾 详细结果保存到: {results_file}")
    
    return generated_results

if __name__ == "__main__":
    results = test_single_generation()