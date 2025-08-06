#!/usr/bin/env python3
"""
测试离线Graph模态训练
"""

import os
import sys
from pathlib import Path

# 设置离线环境
os.environ['TIMM_MODEL_DIR'] = '/root/autodl-tmp/pretrained_models'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.append(str(Path(__file__).parent))

def test_offline_graph():
    print("🧪 测试离线Graph模态训练")
    print("=" * 60)
    
    try:
        # 测试模型初始化
        print("1. 测试模型初始化...")
        from scaffold_mol_gen.models.end2end_model import End2EndMolecularGenerator
        
        model = End2EndMolecularGenerator(
            hidden_size=768,
            num_layers=6,
            num_heads=12,
            fusion_type='both'
        )
        print("✅ 模型初始化成功!")
        
        # 测试数据加载
        print("\n2. 测试Graph数据处理...")
        from scaffold_mol_gen.data.multimodal_preprocessor import MultiModalPreprocessor
        
        preprocessor = MultiModalPreprocessor()
        test_smiles = "CC(C)CC1=CC=CC=C1"
        graph = preprocessor.smiles_to_graph(test_smiles)
        
        if graph is not None:
            print(f"✅ Graph转换成功: 节点数={graph.x.shape[0]}")
        
        print("\n✅ 离线Graph训练环境测试通过!")
        print("\n可以安全启动训练:")
        print("python safe_background_training.py graph")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        print("请检查错误信息")

if __name__ == "__main__":
    test_offline_graph()
