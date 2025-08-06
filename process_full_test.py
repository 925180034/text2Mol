#!/usr/bin/env python3
"""
处理完整test集，生成三种模态数据
"""
import os
import sys
import torch
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from torch_geometric.data import Data
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/root/text2Mol/scaffold-mol-generation')

def smiles_to_graph(smiles):
    """将SMILES转换为图数据"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # 原子特征
        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(), 
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                atom.GetTotalNumHs(),
                atom.GetNumRadicalElectrons(),
                int(atom.IsInRing()),
                int(atom.GetChiralTag()),
                atom.GetMass()
            ]
            atom_features.append(features)
        
        # 边信息
        edge_indices = []
        edge_attrs = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.extend([[i, j], [j, i]])
            
            bond_features = [
                bond.GetBondTypeAsDouble(),
                int(bond.GetIsAromatic()),
                int(bond.GetIsConjugated()),
                int(bond.IsInRing()),
                int(bond.GetStereo())
            ]
            edge_attrs.extend([bond_features, bond_features])
        
        if len(edge_indices) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 5), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        
        x = torch.tensor(atom_features, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    except Exception as e:
        print(f"Error converting SMILES to graph: {e}")
        return None

def smiles_to_image(smiles, size=(299, 299)):
    """将SMILES转换为图像"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        img = Draw.MolToImage(mol, size=size)
        img_array = np.array(img)
        
        # 确保是RGB格式
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        
        # 转换为CHW格式 (3, 299, 299)
        img_array = np.transpose(img_array, (2, 0, 1))
        
        return img_array
    except Exception as e:
        print(f"Error converting SMILES to image: {e}")
        return None

def extract_scaffold(smiles):
    """提取分子的Murcko scaffold"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except:
        return None

def process_full_test_data():
    """处理完整的test数据集"""
    print("\n" + "="*70)
    print("🔬 处理完整Test数据集 - 生成三种模态")
    print("="*70)
    
    # 设置路径
    data_dir = Path('/root/text2Mol/scaffold-mol-generation/Datasets')
    output_dir = data_dir
    
    # 创建输出目录
    (output_dir / 'graph').mkdir(exist_ok=True)
    (output_dir / 'image').mkdir(exist_ok=True)
    
    # 读取完整test数据
    test_file = data_dir / 'test.csv'
    print(f"\n📊 读取test数据: {test_file}")
    df = pd.read_csv(test_file)
    print(f"  样本总数: {len(df)}")
    
    # 检查并添加scaffold列
    if 'scaffold' not in df.columns:
        print("\n⚙️ 提取Scaffold...")
        df['scaffold'] = df['SMILES'].apply(extract_scaffold)
        valid_scaffolds = df['scaffold'].notna().sum()
        print(f"  有效scaffold: {valid_scaffolds}/{len(df)} ({100*valid_scaffolds/len(df):.1f}%)")
    
    # 准备数据容器
    graph_data = []
    image_data = []
    
    print("\n🔄 转换数据为三种模态...")
    
    # 处理每个样本
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理样本"):
        cid = row['CID']
        smiles = row['SMILES']
        scaffold = row.get('scaffold', None)
        text = row.get('text', '') if 'text' in row else row.get('description', '')
        
        # 创建数据字典
        sample_data = {
            'cid': cid,
            'smiles': smiles,
            'scaffold': scaffold,
            'text': text
        }
        
        # 转换SMILES为Graph
        smiles_graph = smiles_to_graph(smiles)
        if smiles_graph is not None:
            sample_data['smiles_graph'] = smiles_graph
        
        # 转换Scaffold为Graph
        if pd.notna(scaffold):
            scaffold_graph = smiles_to_graph(scaffold)
            if scaffold_graph is not None:
                sample_data['scaffold_graph'] = scaffold_graph
        
        graph_data.append(sample_data)
        
        # 转换SMILES为Image
        smiles_image = smiles_to_image(smiles)
        if smiles_image is not None:
            sample_data_img = sample_data.copy()
            sample_data_img['smiles_image'] = smiles_image
        
        # 转换Scaffold为Image
        if pd.notna(scaffold):
            scaffold_image = smiles_to_image(scaffold)
            if scaffold_image is not None:
                sample_data_img['scaffold_image'] = scaffold_image
        
        image_data.append(sample_data_img)
    
    # 保存处理后的数据
    print("\n💾 保存处理后的数据...")
    
    # 保存CSV（包含scaffold）
    csv_output = output_dir / 'test_with_scaffold.csv'
    df.to_csv(csv_output, index=False)
    print(f"  CSV保存到: {csv_output}")
    
    # 保存Graph数据
    graph_output = output_dir / 'graph' / 'test_graphs.pkl'
    with open(graph_output, 'wb') as f:
        pickle.dump(graph_data, f)
    print(f"  Graph数据保存到: {graph_output} ({len(graph_data)} 样本)")
    
    # 保存Image数据
    image_output = output_dir / 'image' / 'test_images.pkl'
    with open(image_output, 'wb') as f:
        pickle.dump(image_data, f)
    print(f"  Image数据保存到: {image_output} ({len(image_data)} 样本)")
    
    # 统计信息
    print("\n📈 数据统计:")
    print(f"  总样本数: {len(df)}")
    print(f"  有效SMILES: {df['SMILES'].notna().sum()}")
    print(f"  有效Scaffold: {df['scaffold'].notna().sum() if 'scaffold' in df.columns else 0}")
    
    # 检查Graph转换成功率
    graph_success = sum(1 for d in graph_data if 'smiles_graph' in d)
    print(f"  Graph转换成功: {graph_success}/{len(df)} ({100*graph_success/len(df):.1f}%)")
    
    # 检查Image转换成功率
    image_success = sum(1 for d in image_data if 'smiles_image' in d)
    print(f"  Image转换成功: {image_success}/{len(df)} ({100*image_success/len(df):.1f}%)")
    
    print("\n✅ 数据处理完成！")
    print("="*70)
    
    return df, graph_data, image_data

if __name__ == "__main__":
    df, graph_data, image_data = process_full_test_data()