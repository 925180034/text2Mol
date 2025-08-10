#!/usr/bin/env python3
"""测试简单案例"""

import sys
sys.path.append('.')

from run_fully_fixed_test import FullyFixedEvaluator
import warnings
warnings.filterwarnings('ignore')

# 创建评估器
evaluator = FullyFixedEvaluator(device='cuda')

# 测试数据
test_smiles = ["CCO", "CC(=O)O", "c1ccccc1"]
test_texts = ["An alcohol", "An acid", "Benzene"]

print("="*60)
print("测试 SMILES → SMILES")
print("="*60)
result1 = evaluator.test_single_combination(
    scaffold_modality='smiles',
    output_modality='smiles',
    test_smiles=test_smiles,
    test_texts=test_texts
)

print("\n" + "="*60)
print("测试 IMAGE → SMILES")
print("="*60)
result2 = evaluator.test_single_combination(
    scaffold_modality='image',
    output_modality='smiles',
    test_smiles=test_smiles,
    test_texts=test_texts
)

print("\n完成！")