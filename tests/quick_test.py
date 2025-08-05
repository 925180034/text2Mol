#!/usr/bin/env python3
"""
快速测试脚本 - 测试模型是否正常工作
"""

import sys
sys.path.append('/root/text2Mol/scaffold-mol-generation')

from inference import MolecularInference
import pandas as pd
from pathlib import Path

def quick_test():
    """快速测试模型功能"""
    
    print("=== 快速模型测试 ===\n")
    
    # 检查模型文件
    model_path = '/root/autodl-tmp/safe_fast_checkpoints/best_model.pt'
    if not Path(model_path).exists():
        print(f"错误：模型文件不存在: {model_path}")
        return
    
    print(f"✓ 找到模型文件: {model_path}")
    
    # 初始化模型
    try:
        print("\n初始化模型...")
        inference = MolecularInference(model_path)
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return
    
    # 测试简单例子
    print("\n=== 测试示例 ===")
    
    test_case = {
        "scaffold": "c1ccccc1",  # 苯环
        "text": "A molecule with anti-inflammatory properties"
    }
    
    print(f"\n测试输入:")
    print(f"- 骨架: {test_case['scaffold']}")
    print(f"- 描述: {test_case['text']}")
    
    try:
        generated = inference.generate_molecule(
            scaffold=test_case['scaffold'],
            text_description=test_case['text'],
            scaffold_type='smiles'
        )
        print(f"- 生成: {generated}")
        print("✓ 生成成功!")
        
        # 保存可视化
        inference.visualize_result(
            test_case['scaffold'],
            generated,
            save_path="quick_test_result.png"
        )
        print("✓ 可视化已保存: quick_test_result.png")
        
    except Exception as e:
        print(f"✗ 生成失败: {e}")
        return
    
    print("\n=== 测试完成 ===")
    print("\n如果看到生成的SMILES，说明模型工作正常！")
    print("现在可以运行完整评估：")
    print("  chmod +x run_evaluation.sh")
    print("  ./run_evaluation.sh")

if __name__ == '__main__':
    quick_test()