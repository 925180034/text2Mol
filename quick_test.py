#!/usr/bin/env python3
"""
快速测试基本功能
"""

import torch
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from scaffold_mol_gen.models.end2end_model import End2EndMolecularGenerator

def test_basic_functionality():
    """测试基本功能"""
    print("🧪 测试基本功能...")
    
    device = 'cpu'
    
    # 创建模型
    print("创建模型...")
    model = End2EndMolecularGenerator(
        hidden_size=768,
        molt5_path="laituan245/molt5-base",
        device=device
    )
    model.eval()
    
    # 测试SMILES输入
    print("测试SMILES输入...")
    try:
        with torch.no_grad():
            output = model.generate(
                scaffold_data="c1ccccc1",
                text_data="Simple aromatic compound",
                scaffold_modality='smiles',
                output_modality='smiles',
                num_beams=1,
                max_length=32,
                num_return_sequences=1
            )
        print(f"✅ SMILES生成成功: {output[0] if output else 'None'}")
    except Exception as e:
        print(f"❌ SMILES生成失败: {e}")
    
    # 测试图生成
    print("测试图生成...")
    try:
        with torch.no_grad():
            graphs = model.generate(
                scaffold_data="c1ccccc1",
                text_data="Simple aromatic compound",
                scaffold_modality='smiles',
                output_modality='graph',
                num_beams=1,
                max_length=32,
                num_return_sequences=1
            )
        if graphs and len(graphs) > 0:
            graph = graphs[0]
            print(f"✅ 图生成成功: {graph.x.shape[0]}个节点, {graph.edge_index.shape[1]}条边")
        else:
            print("❌ 图生成失败: 输出为空")
    except Exception as e:
        print(f"❌ 图生成失败: {e}")
    
    # 测试图像生成
    print("测试图像生成...")
    try:
        with torch.no_grad():
            images = model.generate(
                scaffold_data="c1ccccc1",
                text_data="Simple aromatic compound",
                scaffold_modality='smiles',
                output_modality='image',
                num_beams=1,
                max_length=32,
                num_return_sequences=1
            )
        if images and len(images) > 0:
            print(f"✅ 图像生成成功: {len(images)}张图像")
        else:
            print("❌ 图像生成失败: 输出为空")
    except Exception as e:
        print(f"❌ 图像生成失败: {e}")
    
    print("\n🎯 基本功能测试完成！")

if __name__ == "__main__":
    test_basic_functionality()