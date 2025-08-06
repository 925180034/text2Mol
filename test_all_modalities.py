#!/usr/bin/env python3
"""
测试所有三种模态的训练
验证SMILES, Graph, Image三种输入模态都可以正常工作
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
from scaffold_mol_gen.models.end2end_model import End2EndMolecularGenerator
from scaffold_mol_gen.data.multimodal_dataset import create_data_loaders

def test_modality(modality_name):
    """测试特定模态"""
    print(f"\n=== 测试 {modality_name.upper()} 模态 ===")
    
    try:
        # 1. 创建数据加载器
        train_loader, val_loader, test_loader = create_data_loaders(
            train_csv="Datasets/train.csv",
            val_csv="Datasets/validation.csv",
            test_csv="Datasets/test.csv",
            batch_size=2,
            num_workers=0,
            scaffold_modality=modality_name,  # 指定模态
            max_text_length=128,
            max_smiles_length=128
        )
        print(f"✅ {modality_name} 数据加载器创建成功")
        
        # 2. 创建模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = End2EndMolecularGenerator(device=device)
        print(f"✅ 模型创建成功")
        
        # 3. 测试一个批次
        batch = next(iter(train_loader))
        
        # 获取输入数据
        if modality_name == 'smiles':
            scaffold_input = batch['scaffold_data']  # SMILES字符串
        elif modality_name == 'graph':
            scaffold_input = batch['scaffold_data']  # 应该是图数据
        elif modality_name == 'image':
            scaffold_input = batch['scaffold_data']  # 应该是图像数据
        
        text_input = batch['text_data']
        target_smiles = batch['target_smiles']
        
        print(f"✅ 批次数据获取成功")
        print(f"   - 批次大小: {len(text_input)}")
        print(f"   - Scaffold类型: {type(scaffold_input)}")
        if isinstance(scaffold_input, list):
            print(f"   - Scaffold示例: {scaffold_input[0]}")
        
        # 4. 前向传播测试
        try:
            outputs = model(
                scaffold_input=scaffold_input[:1],  # 只用第一个样本
                text_input=text_input[:1],
                scaffold_modality=modality_name
            )
            print(f"✅ 前向传播成功")
            print(f"   - 输出形状: {outputs.shape}")
            
            # 5. 生成测试
            generated = model.generate(
                scaffold_input=scaffold_input[:1],
                text_input=text_input[:1],
                scaffold_modality=modality_name,
                max_length=50  # 短一点避免超时
            )
            print(f"✅ 生成测试成功")
            print(f"   - 生成结果: {generated[0][:50]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ 前向传播失败: {str(e)}")
            return False
            
    except Exception as e:
        print(f"❌ {modality_name} 模态测试失败: {str(e)}")
        return False

def main():
    print("=== 多模态测试 ===")
    print("测试所有三种输入模态是否都可以正常工作")
    
    # 测试三种模态
    modalities = ['smiles', 'graph', 'image']
    results = {}
    
    for modality in modalities:
        results[modality] = test_modality(modality)
    
    # 总结结果
    print("\n=== 测试结果总结 ===")
    for modality, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{modality.upper()} 模态: {status}")
    
    # 解释多模态数据
    print("\n=== 多模态数据说明 ===")
    print("• 原始数据集只包含SMILES格式的分子结构")
    print("• Graph模态: 系统自动将SMILES转换为分子图结构")
    print("• Image模态: 系统自动将SMILES转换为2D分子图像")
    print("• 所有模态表示的都是同一个分子，只是编码方式不同")
    print("• 模型需要通过训练学习不同模态之间的关系")
    
    successful_count = sum(results.values())
    print(f"\n总体成功率: {successful_count}/{len(modalities)} ({successful_count/len(modalities)*100:.1f}%)")

if __name__ == "__main__":
    main()