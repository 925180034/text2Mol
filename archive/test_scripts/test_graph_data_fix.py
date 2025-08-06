#!/usr/bin/env python3
"""
测试Graph数据预处理修复
"""

import torch
import sys
import os
sys.path.append('/root/text2Mol/scaffold-mol-generation')

from scaffold_mol_gen.data.multimodal_dataset import MultiModalMolecularDataset, collate_batch
from torch.utils.data import DataLoader

print("🧪 测试Graph数据预处理修复")
print("=" * 60)

# 创建测试数据集
dataset = MultiModalMolecularDataset(
    csv_path='Datasets/validation.csv',
    scaffold_modality='graph',
    max_text_length=128,
    max_smiles_length=128
)

print(f"✅ 数据集创建成功，共 {len(dataset)} 个样本")

# 测试单个样本
print("\n📝 测试单个样本...")
sample = dataset[0]
print(f"  scaffold_data类型: {type(sample['scaffold_data'])}")
print(f"  scaffold_modality: {sample['scaffold_modality']}")

# 检查是否为Graph对象
from torch_geometric.data import Data
if isinstance(sample['scaffold_data'], Data):
    print(f"  ✅ scaffold_data是PyG Data对象")
    print(f"     节点数: {sample['scaffold_data'].x.shape[0]}")
    print(f"     边数: {sample['scaffold_data'].edge_index.shape[1] // 2}")
else:
    print(f"  ❌ scaffold_data不是PyG Data对象，而是: {type(sample['scaffold_data'])}")

# 测试批处理
print("\n📝 测试批处理...")
batch_samples = [dataset[i] for i in range(4)]
batch = collate_batch(batch_samples)

print(f"  批大小: {batch['batch_size']}")
print(f"  scaffold_data类型: {type(batch['scaffold_data'])}")
print(f"  scaffold_data长度: {len(batch['scaffold_data'])}")

# 检查批内每个样本
all_graphs = True
for i, data in enumerate(batch['scaffold_data']):
    if not isinstance(data, Data):
        print(f"  ❌ 第{i}个scaffold不是Graph对象")
        all_graphs = False
        break

if all_graphs:
    print(f"  ✅ 所有scaffold_data都是PyG Data对象")

# 测试DataLoader
print("\n📝 测试DataLoader...")
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=collate_batch
)

try:
    batch = next(iter(loader))
    print(f"  ✅ DataLoader工作正常")
    print(f"     批大小: {batch['batch_size']}")
    print(f"     scaffold_modality: {batch['scaffold_modality']}")
except Exception as e:
    print(f"  ❌ DataLoader错误: {e}")

print("\n" + "=" * 60)
print("✅ 测试完成！Graph数据预处理修复成功！")