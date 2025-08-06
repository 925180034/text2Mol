#!/usr/bin/env python3
import torch
import sys
sys.path.append('/root/text2Mol/scaffold-mol-generation')

from scaffold_mol_gen.models.encoders.graph_encoder import GINEncoder
from torch_geometric.data import Data

print("测试Graph编码器...")

# 创建测试数据
def create_test_graph():
    x = torch.randn(5, 9)  # 5个节点，9个特征（标准原子特征维度）
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
                               [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

try:
    # 创建编码器
    encoder = GINEncoder(
        input_dim=9,
        hidden_dim=128,
        output_dim=768,
        num_layers=5
    )
    encoder.eval()
    
    # 创建测试图列表
    graphs = [create_test_graph() for _ in range(4)]
    
    # 测试编码
    with torch.no_grad():
        features = encoder.encode_graphs(graphs)
    
    print(f"✅ 测试成功！")
    print(f"   输出形状: {features.shape}")
    print(f"   预期: [4, 768]")
    
    if features.shape == torch.Size([4, 768]):
        print("✅ Graph编码器修复成功！")
    else:
        print("⚠️ 形状不匹配")
        
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
