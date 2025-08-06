#!/usr/bin/env python3
"""
完全修复Graph训练问题
"""

import os
import sys

print("🔧 完全修复Graph训练问题")
print("=" * 60)

# 方案1: 降级PyTorch Geometric（推荐但耗时）
print("\n方案1: 降级PyTorch Geometric到兼容版本")
print("pip install torch-geometric==2.3.1")
print("⚠️ 这需要重新安装，可能需要几分钟")

# 方案2: 修改代码绕过问题
print("\n方案2: 修改代码，使用简单的批处理方法")

fix_code = '''
# 修改 scaffold_mol_gen/models/encoders/graph_encoder.py
# 替换 encode_graphs 方法中的批处理部分

def encode_graphs(self, graph_list):
    """
    编码图列表 - 修复版
    """
    # 手动批处理，避免使用Batch.from_data_list
    
    # 收集所有节点特征
    x_list = []
    edge_index_list = []
    edge_attr_list = []
    batch_list = []
    
    node_offset = 0
    for i, data in enumerate(graph_list):
        # 节点特征
        x_list.append(data.x)
        
        # 边索引（需要偏移）
        edge_index = data.edge_index + node_offset
        edge_index_list.append(edge_index)
        
        # 边特征
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_attr_list.append(data.edge_attr)
        
        # 批次索引
        batch_list.append(torch.full((data.x.size(0),), i, dtype=torch.long))
        
        node_offset += data.x.size(0)
    
    # 拼接
    x = torch.cat(x_list, dim=0)
    edge_index = torch.cat(edge_index_list, dim=1)
    edge_attr = torch.cat(edge_attr_list, dim=0) if edge_attr_list else None
    batch = torch.cat(batch_list, dim=0)
    
    # 移到设备
    x = x.to(self.device)
    edge_index = edge_index.to(self.device)
    if edge_attr is not None:
        edge_attr = edge_attr.to(self.device)
    batch = batch.to(self.device)
    
    # GIN编码
    if edge_attr is not None:
        h = self.gin(x, edge_index, edge_attr)
    else:
        h = self.gin(x, edge_index)
    
    # 全局池化
    graph_features = global_mean_pool(h, batch)
    
    # 投影到输出维度
    graph_features = self.projection(graph_features)
    
    return graph_features
'''

print("\n应用修复...")

# 修改graph_encoder.py
graph_encoder_path = "scaffold_mol_gen/models/encoders/graph_encoder.py"

# 读取文件
with open(graph_encoder_path, 'r') as f:
    content = f.read()

# 检查是否已经修复
if "手动批处理" in content:
    print("✅ 已经应用过修复")
else:
    # 替换encode_graphs方法
    import re
    
    # 找到encode_graphs方法
    pattern = r'def encode_graphs\(self, graph_list\):.*?(?=\n    def|\nclass|\Z)'
    
    replacement = '''def encode_graphs(self, graph_list):
        """
        编码图列表 - 修复版（手动批处理）
        
        Args:
            graph_list: 图数据列表
            
        Returns:
            graph_features: [batch_size, hidden_size]
        """
        import torch
        from torch_geometric.nn import global_mean_pool
        
        # 手动批处理，避免使用Batch.from_data_list
        
        # 收集所有节点特征
        x_list = []
        edge_index_list = []
        edge_attr_list = []
        batch_list = []
        
        node_offset = 0
        for i, data in enumerate(graph_list):
            # 节点特征
            x_list.append(data.x)
            
            # 边索引（需要偏移）
            edge_index = data.edge_index + node_offset
            edge_index_list.append(edge_index)
            
            # 边特征
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                edge_attr_list.append(data.edge_attr)
            
            # 批次索引
            batch_list.append(torch.full((data.x.size(0),), i, dtype=torch.long))
            
            node_offset += data.x.size(0)
        
        # 拼接
        x = torch.cat(x_list, dim=0)
        edge_index = torch.cat(edge_index_list, dim=1)
        edge_attr = torch.cat(edge_attr_list, dim=0) if edge_attr_list else None
        batch = torch.cat(batch_list, dim=0)
        
        # 移到设备
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        if edge_attr is not None:
            edge_attr = edge_attr.to(self.device)
        batch = batch.to(self.device)
        
        # GIN编码
        if edge_attr is not None:
            h = self.gin(x, edge_index, edge_attr)
        else:
            h = self.gin(x, edge_index)
        
        # 全局池化
        graph_features = global_mean_pool(h, batch)
        
        # 投影到输出维度
        graph_features = self.projection(graph_features)
        
        return graph_features'''
    
    # 替换
    content_new = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # 写回文件
    with open(graph_encoder_path, 'w') as f:
        f.write(content_new)
    
    print("✅ 修复已应用到graph_encoder.py")

print("\n重新启动Graph训练...")

# 杀死之前的进程
os.system("pkill -f 'train_multimodal.*graph'")
time.sleep(2)

# 重启训练
cmd = """
CUDA_VISIBLE_DEVICES=1 python train_multimodal.py \
    --scaffold-modality graph \
    --batch-size 8 \
    --epochs 1 \
    --lr 2e-5 \
    --output-dir /root/autodl-tmp/text2Mol-outputs/fast_training/graph \
    > logs/graph_train_final.log 2>&1 &
"""

os.system(cmd)
print("✅ Graph训练已重启")
print("\n查看日志: tail -f logs/graph_train_final.log")

import time
time.sleep(10)

# 检查是否成功
os.system("tail -20 logs/graph_train_final.log | grep -E '(Epoch|loss|ERROR)'")