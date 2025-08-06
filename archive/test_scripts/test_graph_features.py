#!/usr/bin/env python3
"""
测试Graph特征维度
"""

import torch
import sys
sys.path.append('/root/text2Mol/scaffold-mol-generation')

from scaffold_mol_gen.data.multimodal_preprocessor import MultiModalPreprocessor

print("🧪 测试Graph特征维度")
print("=" * 60)

preprocessor = MultiModalPreprocessor()

# 测试SMILES
test_smiles = "CC(C)c1ccc(cc1)C(C)C"

# 转换为Graph
graph = preprocessor.smiles_to_graph(test_smiles)

if graph is not None:
    print(f"✅ Graph创建成功")
    print(f"  节点数: {graph.x.shape[0]}")
    print(f"  节点特征维度: {graph.x.shape[1]}")
    print(f"  边数: {graph.edge_index.shape[1] // 2}")
    
    if graph.x.shape[1] == 9:
        print(f"  ✅ 节点特征维度正确 (9维)")
    else:
        print(f"  ❌ 节点特征维度错误，期望9维，实际{graph.x.shape[1]}维")
    
    # 显示第一个节点的特征
    print(f"\n第一个节点的特征:")
    features = graph.x[0].tolist()
    feature_names = [
        "原子序数", "度数", "形式电荷", "杂化类型", 
        "芳香性", "氢原子数", "自由基电子", "在环中", "手性"
    ]
    for name, val in zip(feature_names, features):
        print(f"  {name}: {val}")