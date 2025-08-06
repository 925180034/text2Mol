#!/usr/bin/env python3
"""
测试Graph编码器是否工作
"""

import torch
import sys
sys.path.append('/root/text2Mol/scaffold-mol-generation')

from scaffold_mol_gen.models.encoders.graph_encoder import GraphEncoder
from torch_geometric.data import Data

# 创建测试数据
def create_test_graph():
    x = torch.randn(5, 6)  # 5个节点，6个特征
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                               [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
    edge_attr = torch.randn(8, 3)  # 8条边，3个特征
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

print("测试Graph编码器...")

try:
    # 创建编码器
    encoder = GraphEncoder(
        input_dim=6,
        hidden_dim=128,
        output_dim=768,
        num_layers=5,
        bond_features=True
    )
    encoder.eval()
    
    # 创建测试图列表
    graphs = [create_test_graph() for _ in range(4)]
    
    # 测试编码
    with torch.no_grad():
        features = encoder.encode_graphs(graphs)
    
    print(f"✅ 测试成功！输出形状: {features.shape}")
    print(f"   预期: [4, 768], 实际: {list(features.shape)}")
    
    if features.shape == torch.Size([4, 768]):
        print("✅ 形状正确！Graph编码器修复成功！")
    else:
        print("⚠️ 形状不匹配，可能还有问题")
        
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
