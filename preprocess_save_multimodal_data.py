#\!/usr/bin/env python3
"""
预处理数据集，生成并保存Graph和Image格式
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import torch
from torch_geometric.data import Data
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/root/text2Mol/scaffold-mol-generation')

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
    
    # 创建图数据
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float).view(-1, 1)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def smiles_to_image(smiles, size=(299, 299)):
    """将SMILES转换为图像"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # 生成分子图像
    img = Draw.MolToImage(mol, size=size)
    img_array = np.array(img)
    
    return img_array

def process_dataset(csv_path, output_dir, dataset_name):
    """处理单个数据集"""
    print(f"\n📊 处理 {dataset_name} 数据集...")
    
    # 读取CSV
    df = pd.read_csv(csv_path)
    print(f"  样本数: {len(df)}")
    
    # 创建输出目录
    graph_dir = output_dir / 'graph' / dataset_name
    image_dir = output_dir / 'image' / dataset_name
    graph_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理每个样本
    graph_data_list = []
    image_data_list = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"处理{dataset_name}"):
        cid = row['CID']
        scaffold = row['scaffold']
        smiles = row['SMILES']
        
        # 转换scaffold为图
        scaffold_graph = smiles_to_graph(scaffold)
        if scaffold_graph is not None:
            scaffold_graph.cid = cid
            scaffold_graph.scaffold = scaffold
            scaffold_graph.smiles = smiles
            graph_data_list.append(scaffold_graph)
        
        # 转换scaffold为图像
        scaffold_image = smiles_to_image(scaffold)
        if scaffold_image is not None:
            image_data = {
                'cid': cid,
                'scaffold': scaffold,
                'smiles': smiles,
                'image': scaffold_image
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
    print("🚀 开始预处理多模态数据集...")
    
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
            n_graphs, n_images = process_dataset(csv_path, datasets_dir, name)
            total_graphs += n_graphs
            total_images += n_images
    
    print(f"\n✅ 预处理完成\!")
    print(f"  总计图数据: {total_graphs}")
    print(f"  总计图像数据: {total_images}")
    print(f"  数据保存在: {datasets_dir}/graph/ 和 {datasets_dir}/image/")

if __name__ == "__main__":
    main()
