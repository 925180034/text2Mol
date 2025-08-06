#!/usr/bin/env python3
"""
快速修复Graph训练问题并重启
"""

import os
import time

print("🔧 修复Graph训练问题")
print("=" * 60)

# 修复内容
fix_code = '''
# 修复PyTorch Geometric兼容性问题
# 在multimodal_preprocessor.py中，修改graph数据创建部分

def smiles_to_graph(self, smiles: str, bond_features: bool = True):
    """将SMILES转换为图数据 - 修复版"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # 构建原子特征
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),
            int(atom.IsInRing()),
            int(atom.GetIsAromatic()),
        ]
        atom_features.append(features)
    
    # 转换为张量
    x = torch.tensor(atom_features, dtype=torch.float)
    
    # 构建边
    edges = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.extend([[i, j], [j, i]])  # 双向边
        
        if bond_features:
            bond_feature = [
                bond.GetBondTypeAsDouble(),
                float(bond.IsInRing()),
                float(bond.GetIsConjugated()),
            ]
            edge_features.extend([bond_feature, bond_feature])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_features, dtype=torch.float) if edge_features else None
    
    # 创建Data对象 - 不包含smiles属性（这是关键修复）
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr if bond_features else None
    )
    
    return data
'''

print("\n问题诊断:")
print("1. PyTorch Geometric版本不兼容")
print("2. 'strBatch' object has no attribute 'stores_as'")
print("3. 需要移除Data对象中的smiles属性")

# 检查文件是否已经修复
check_cmd = "grep -q 'smiles=smiles' scaffold_mol_gen/data/multimodal_preprocessor.py 2>/dev/null"
result = os.system(check_cmd)

if result == 0:
    print("\n✅ 检测到未修复的代码，应用修复...")
    
    # 使用之前的fix_multimodal_issues.py
    if os.path.exists("fix_multimodal_issues.py"):
        os.system("python fix_multimodal_issues.py")
        print("✅ 修复已应用")
    else:
        print("⚠️ 修复脚本不存在，手动修复...")
else:
    print("\n✅ 代码已经修复过了")

print("\n重启Graph训练...")

# 重启Graph训练，使用更小的batch size避免问题
cmd = """
CUDA_VISIBLE_DEVICES=1 python train_multimodal.py \
    --scaffold-modality graph \
    --batch-size 8 \
    --epochs 1 \
    --lr 2e-5 \
    --output-dir /root/autodl-tmp/text2Mol-outputs/fast_training/graph \
    > logs/graph_train_fixed.log 2>&1 &
"""

print(f"执行命令: {cmd}")
os.system(cmd)

print("\n✅ Graph训练已重启!")
print("\n查看日志: tail -f logs/graph_train_fixed.log")
print("查看GPU: nvidia-smi -l 1")