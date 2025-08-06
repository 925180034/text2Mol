#\!/usr/bin/env python3
"""
完整的数据预处理脚本：生成scaffold、Graph和Image格式
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw, Scaffolds
from PIL import Image
import torch
from torch_geometric.data import Data
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/root/text2Mol/scaffold-mol-generation')

def get_murcko_scaffold(smiles):
    """获取Murcko scaffold"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles  # 如果失败，返回原始SMILES
    scaffold = Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)

def smiles_to_graph(smiles):
    """将SMILES转换为图数据"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # 获取原子特征
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic())
        ]
        atom_features.append(features)
    
    # 获取边信息
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.extend([[i, j], [j, i]])
        bond_type = bond.GetBondTypeAsDouble()
        edge_attrs.extend([bond_type, bond_type])
    
    if len(edge_indices) == 0:  # 单原子分子
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float).view(-1, 1)
    
    # 创建图数据
    x = torch.tensor(atom_features, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def smiles_to_image(smiles, size=(299, 299)):
    """将SMILES转换为图像"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # 返回空白图像
        return np.ones((size[0], size[1], 3), dtype=np.uint8) * 255
    
    # 生成分子图像
    img = Draw.MolToImage(mol, size=size)
    img_array = np.array(img)
    
    # 确保是RGB格式
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    
    return img_array

def process_and_save_dataset(csv_path, output_dir, dataset_name):
    """处理并保存单个数据集"""
    print(f"\n📊 处理 {dataset_name} 数据集...")
    
    # 读取CSV
    df = pd.read_csv(csv_path)
    print(f"  原始样本数: {len(df)}")
    
    # 生成scaffold
    print("  生成scaffold...")
    scaffolds = []
    for smiles in tqdm(df['SMILES'], desc="生成scaffold"):
        scaffold = get_murcko_scaffold(smiles)
        scaffolds.append(scaffold)
    df['scaffold'] = scaffolds
    
    # 使用description作为text
    df['text'] = df['description'].fillna('')
    
    # 保存增强的CSV（包含scaffold）
    enhanced_csv = output_dir / f'{dataset_name}_with_scaffold.csv'
    df[['CID', 'scaffold', 'text', 'SMILES']].to_csv(enhanced_csv, index=False)
    print(f"  ✅ 保存增强CSV: {enhanced_csv}")
    
    # 创建输出目录
    graph_dir = output_dir / 'graph' / dataset_name
    image_dir = output_dir / 'image' / dataset_name
    graph_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理每个样本
    graph_data_list = []
    image_data_list = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"转换{dataset_name}"):
        cid = row['CID']
        scaffold = row['scaffold']
        smiles = row['SMILES']
        text = row['text']
        
        # 转换SMILES为图（用于完整分子）
        smiles_graph = smiles_to_graph(smiles)
        # 转换scaffold为图
        scaffold_graph = smiles_to_graph(scaffold)
        
        if smiles_graph is not None and scaffold_graph is not None:
            graph_data = {
                'cid': cid,
                'scaffold': scaffold,
                'scaffold_graph': scaffold_graph,
                'smiles': smiles,
                'smiles_graph': smiles_graph,
                'text': text
            }
            graph_data_list.append(graph_data)
        
        # 转换SMILES为图像
        smiles_image = smiles_to_image(smiles)
        # 转换scaffold为图像
        scaffold_image = smiles_to_image(scaffold)
        
        image_data = {
            'cid': cid,
            'scaffold': scaffold,
            'scaffold_image': scaffold_image,
            'smiles': smiles,
            'smiles_image': smiles_image,
            'text': text
        }
        image_data_list.append(image_data)
    
    # 保存图数据
    graph_path = graph_dir / f'{dataset_name}_graphs.pkl'
    with open(graph_path, 'wb') as f:
        pickle.dump(graph_data_list, f)
    print(f"  ✅ 保存图数据: {graph_path} ({len(graph_data_list)} 个)")
    
    # 保存图像数据
    image_path = image_dir / f'{dataset_name}_images.pkl'
    with open(image_path, 'wb') as f:
        pickle.dump(image_data_list, f)
    print(f"  ✅ 保存图像数据: {image_path} ({len(image_data_list)} 个)")
    
    return len(graph_data_list), len(image_data_list)

def main():
    print("🚀 开始完整的数据预处理...")
    print("  将生成: scaffold, Graph格式, Image格式")
    
    # 设置路径
    base_dir = Path('/root/text2Mol/scaffold-mol-generation')
    datasets_dir = base_dir / 'Datasets'
    
    # 处理三个数据集
    datasets = [
        ('train.csv', 'train'),
        ('validation.csv', 'validation'),
        ('test.csv', 'test')
    ]
    
    total_graphs = 0
    total_images = 0
    
    for csv_file, name in datasets:
        csv_path = datasets_dir / csv_file
        if csv_path.exists():
            n_graphs, n_images = process_and_save_dataset(csv_path, datasets_dir, name)
            total_graphs += n_graphs
            total_images += n_images
        else:
            print(f"  ⚠️ 文件不存在: {csv_path}")
    
    print(f"\n✅ 数据预处理完成\!")
    print(f"  总计图数据: {total_graphs}")
    print(f"  总计图像数据: {total_images}")
    print(f"  数据保存位置:")
    print(f"    - 增强CSV: {datasets_dir}/*_with_scaffold.csv")
    print(f"    - 图数据: {datasets_dir}/graph/")
    print(f"    - 图像数据: {datasets_dir}/image/")

if __name__ == "__main__":
    main()
