#!/usr/bin/env python3
"""
检查和演示多模态数据的生成
展示如何从SMILES生成Graph和Image数据
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import torch
import matplotlib.pyplot as plt
from scaffold_mol_gen.data.multimodal_preprocessor import MultiModalPreprocessor

def check_multimodal_data():
    print("=== 多模态数据检查 ===\n")
    
    # 1. 读取数据集
    print("1. 读取原始数据集...")
    df = pd.read_csv("Datasets/train.csv")
    print(f"   数据集大小: {len(df)} 条记录")
    print(f"   列名: {list(df.columns)}")
    print(f"   数据集中没有直接的graph或image列，需要动态生成\n")
    
    # 2. 初始化预处理器
    print("2. 初始化多模态预处理器...")
    preprocessor = MultiModalPreprocessor()
    
    # 3. 选择一个样本进行演示
    sample_idx = 0
    sample = df.iloc[sample_idx]
    smiles = sample['SMILES']
    description = sample['description']
    
    print(f"\n3. 样本 {sample_idx}:")
    print(f"   SMILES: {smiles}")
    print(f"   描述: {description[:100]}...")
    
    # 4. 生成Graph数据
    print("\n4. 生成Graph数据...")
    graph_data = preprocessor.smiles_to_graph(smiles)
    if graph_data:
        print(f"   ✅ Graph生成成功!")
        print(f"   - 节点数: {graph_data.x.shape[0]}")
        print(f"   - 节点特征维度: {graph_data.x.shape[1]}")
        print(f"   - 边数: {graph_data.edge_index.shape[1]}")
        if graph_data.edge_attr is not None:
            print(f"   - 边特征维度: {graph_data.edge_attr.shape[1]}")
    else:
        print("   ❌ Graph生成失败!")
    
    # 5. 生成Image数据
    print("\n5. 生成Image数据...")
    image_data = preprocessor.smiles_to_image(smiles)
    if image_data is not None:
        print(f"   ✅ Image生成成功!")
        print(f"   - 图像尺寸: {image_data.shape}")
        print(f"   - 数据类型: {image_data.dtype}")
        print(f"   - 值范围: [{image_data.min()}, {image_data.max()}]")
        
        # 保存示例图像
        plt.figure(figsize=(6, 6))
        plt.imshow(image_data)
        plt.axis('off')
        plt.title(f"分子图像示例: {smiles[:30]}...")
        plt.savefig("experiments/sample_molecule_image.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("   - 示例图像已保存到: experiments/sample_molecule_image.png")
    else:
        print("   ❌ Image生成失败!")
    
    # 6. 批量处理示例
    print("\n6. 批量处理示例（前10个样本）...")
    success_graph = 0
    success_image = 0
    
    for i in range(min(10, len(df))):
        smiles = df.iloc[i]['SMILES']
        
        # 尝试生成graph
        graph = preprocessor.smiles_to_graph(smiles)
        if graph:
            success_graph += 1
        
        # 尝试生成image
        image = preprocessor.smiles_to_image(smiles)
        if image is not None:
            success_image += 1
    
    print(f"   Graph生成成功率: {success_graph}/10 ({success_graph*10}%)")
    print(f"   Image生成成功率: {success_image}/10 ({success_image*10}%)")
    
    # 7. 总结
    print("\n=== 总结 ===")
    print("• 原始数据集只包含SMILES和文本描述")
    print("• Graph数据通过MultiModalPreprocessor.smiles_to_graph()动态生成")
    print("• Image数据通过MultiModalPreprocessor.smiles_to_image()动态生成")
    print("• 模型训练时会根据scaffold_modality参数动态转换数据")
    print("• 三种模态（SMILES, Graph, Image）共享相同的分子结构，只是表示形式不同")
    
    # 8. 如何使用不同模态
    print("\n=== 如何使用不同模态 ===")
    print("训练时指定scaffold_modality参数：")
    print("  --scaffold-modality smiles  # 使用SMILES模态（默认）")
    print("  --scaffold-modality graph   # 使用Graph模态（动态生成）")
    print("  --scaffold-modality image   # 使用Image模态（动态生成）")

if __name__ == "__main__":
    check_multimodal_data()