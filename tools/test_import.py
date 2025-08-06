#!/usr/bin/env python3
"""
测试导入问题
"""

import sys
import time

print("测试导入...")
start = time.time()

try:
    print("1. 导入基础库...")
    import torch
    print(f"   ✅ torch: {torch.__version__}")
    
    print("2. 测试CUDA...")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    print(f"   GPU数量: {torch.cuda.device_count()}")
    
    print("3. 导入项目模块...")
    sys.path.append('/root/text2Mol/scaffold-mol-generation')
    
    # 逐个测试导入
    try:
        from scaffold_mol_gen.models.end2end_model import End2EndMolecularGenerator
        print("   ✅ End2EndMolecularGenerator")
    except Exception as e:
        print(f"   ❌ End2EndMolecularGenerator: {e}")
    
    try:
        from scaffold_mol_gen.data.multimodal_dataset import create_data_loaders
        print("   ✅ create_data_loaders")
    except Exception as e:
        print(f"   ❌ create_data_loaders: {e}")
    
    try:
        from scaffold_mol_gen.training.metrics import GenerationMetrics
        print("   ✅ GenerationMetrics")
    except Exception as e:
        print(f"   ❌ GenerationMetrics: {e}")
    
    try:
        from scaffold_mol_gen.utils.mol_utils import MolecularUtils
        print("   ✅ MolecularUtils")
    except Exception as e:
        print(f"   ❌ MolecularUtils: {e}")
    
    elapsed = time.time() - start
    print(f"\n导入耗时: {elapsed:.1f}秒")
    
except Exception as e:
    print(f"❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()