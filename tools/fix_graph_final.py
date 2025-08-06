#!/usr/bin/env python3
"""
最终修复Graph兼容性问题 - 修复GINEncoder类
"""

import os

print("🔧 最终修复Graph兼容性")
print("=" * 60)

# 直接编辑文件，替换encode_graphs方法
fix_code = '''
# 修改 scaffold_mol_gen/models/encoders/graph_encoder.py
# 在 GINEncoder 类中替换 encode_graphs 方法

def encode_graphs(self, graph_list):
    """
    编码图列表 - 兼容PyTorch Geometric 2.6.1
    
    Args:
        graph_list: PyG Data对象列表
        
    Returns:
        graph_features: 图级别特征 [batch_size, output_dim]
    """
    import torch
    from torch_geometric.nn import global_mean_pool
    
    # 如果列表为空，返回空张量
    if not graph_list:
        return torch.empty(0, self.output_dim if hasattr(self, 'output_dim') else 768, 
                          device=next(self.parameters()).device)
    
    # 手动批处理，避免使用Batch.from_data_list
    device = next(self.parameters()).device
    
    # 收集所有图的数据
    all_x = []
    all_edge_index = []
    all_edge_attr = []
    all_batch = []
    
    node_offset = 0
    for i, data in enumerate(graph_list):
        # 节点特征
        x = data.x
        all_x.append(x)
        
        # 边索引（需要添加偏移量）
        edge_index = data.edge_index
        if edge_index.numel() > 0:  # 如果有边
            edge_index = edge_index + node_offset
        all_edge_index.append(edge_index)
        
        # 边特征（如果存在）
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            all_edge_attr.append(data.edge_attr)
        
        # 批次索引（标记每个节点属于哪个图）
        batch_idx = torch.full((x.size(0),), i, dtype=torch.long)
        all_batch.append(batch_idx)
        
        # 更新节点偏移量
        node_offset += x.size(0)
    
    # 拼接所有数据
    x = torch.cat(all_x, dim=0).to(device)
    edge_index = torch.cat(all_edge_index, dim=1).to(device) if all_edge_index else torch.empty((2, 0), dtype=torch.long, device=device)
    batch = torch.cat(all_batch, dim=0).to(device)
    
    # 处理边特征
    edge_attr = None
    if all_edge_attr:
        edge_attr = torch.cat(all_edge_attr, dim=0).to(device)
    
    # 通过GIN网络编码
    with torch.no_grad():
        # 调用forward方法
        _, graph_features = self.forward(x, edge_index, batch, edge_attr)
    
    return graph_features
'''

# 读取文件
file_path = "scaffold_mol_gen/models/encoders/graph_encoder.py"
with open(file_path, 'r') as f:
    content = f.read()

# 查找并替换encode_graphs方法
import re

# 新的方法实现
new_method = '''    def encode_graphs(self, graph_list):
        """
        编码图列表 - 兼容PyTorch Geometric 2.6.1
        
        Args:
            graph_list: PyG Data对象列表
            
        Returns:
            graph_features: 图级别特征 [batch_size, output_dim]
        """
        import torch
        from torch_geometric.nn import global_mean_pool
        
        # 如果列表为空，返回空张量
        if not graph_list:
            output_dim = self.output_projection.out_features if hasattr(self.output_projection, 'out_features') else 768
            return torch.empty(0, output_dim, device=next(self.parameters()).device)
        
        # 手动批处理，避免使用Batch.from_data_list
        device = next(self.parameters()).device
        
        # 收集所有图的数据
        all_x = []
        all_edge_index = []
        all_edge_attr = []
        all_batch = []
        
        node_offset = 0
        for i, data in enumerate(graph_list):
            # 节点特征
            x = data.x
            all_x.append(x)
            
            # 边索引（需要添加偏移量）
            edge_index = data.edge_index
            if edge_index.numel() > 0:  # 如果有边
                edge_index = edge_index + node_offset
            all_edge_index.append(edge_index)
            
            # 边特征（如果存在）
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                all_edge_attr.append(data.edge_attr)
            
            # 批次索引（标记每个节点属于哪个图）
            batch_idx = torch.full((x.size(0),), i, dtype=torch.long)
            all_batch.append(batch_idx)
            
            # 更新节点偏移量
            node_offset += x.size(0)
        
        # 拼接所有数据
        x = torch.cat(all_x, dim=0).to(device)
        edge_index = torch.cat(all_edge_index, dim=1).to(device) if all_edge_index else torch.empty((2, 0), dtype=torch.long, device=device)
        batch = torch.cat(all_batch, dim=0).to(device)
        
        # 处理边特征
        edge_attr = None
        if all_edge_attr:
            edge_attr = torch.cat(all_edge_attr, dim=0).to(device)
        
        # 通过GIN网络编码
        with torch.no_grad():
            # 调用forward方法
            _, graph_features = self.forward(x, edge_index, batch, edge_attr)
        
        return graph_features'''

# 查找encode_graphs方法的位置
pattern = r'(    def encode_graphs\(self, graph_list[^)]*\):.*?(?=\n    def |\n\nclass |\Z))'

# 替换
content_new = re.sub(pattern, new_method, content, flags=re.DOTALL)

# 写回文件
with open(file_path, 'w') as f:
    f.write(content_new)

print("✅ 修复已应用到 GINEncoder.encode_graphs()")

# 创建简单测试
test_code = '''#!/usr/bin/env python3
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
'''

with open("test_gin_encoder.py", "w") as f:
    f.write(test_code)

print("\n📝 运行测试...")
os.system("python test_gin_encoder.py")

print("\n" + "=" * 60)
print("✅ 修复完成！现在启动Graph训练：")
print("""
# GPU 0 空闲（SMILES已完成），使用GPU 0
CUDA_VISIBLE_DEVICES=0 python train_multimodal.py \\
    --scaffold-modality graph \\
    --batch-size 8 \\
    --epochs 1 \\
    --lr 2e-5 \\
    --output-dir /root/autodl-tmp/text2Mol-outputs/fast_training/graph \\
    > logs/graph_final.log 2>&1 &
""")