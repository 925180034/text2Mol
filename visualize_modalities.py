#!/usr/bin/env python3
"""
多模态数据可视化工具
可视化Graph、Image和SMILES格式的分子数据
"""
import os
import sys
import torch
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

sys.path.append('/root/text2Mol/scaffold-mol-generation')

def load_pkl_data(file_path):
    """加载pkl文件"""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def visualize_graph_data(graph_data, idx=0):
    """可视化Graph数据的统计信息"""
    if idx >= len(graph_data):
        return None
    
    item = graph_data[idx]
    
    # 获取实际的graph数据
    if isinstance(item, dict):
        if 'smiles_graph' in item:
            graph = item['smiles_graph']
        elif 'scaffold_graph' in item:
            graph = item['scaffold_graph']
        else:
            graph = item
    else:
        graph = item
    
    info = {
        'num_atoms': graph.x.shape[0] if hasattr(graph, 'x') else 0,
        'num_bonds': graph.edge_index.shape[1] // 2 if hasattr(graph, 'edge_index') else 0,
        'atom_features_dim': graph.x.shape[1] if hasattr(graph, 'x') else 0,
        'has_edge_attr': hasattr(graph, 'edge_attr')
    }
    
    # 如果有原子特征，显示前几个原子的信息
    if hasattr(graph, 'x') and len(graph.x) > 0:
        info['first_3_atoms'] = graph.x[:3].numpy().tolist()
    
    return info

def visualize_image_data(image_data, idx=0, image_type='smiles'):
    """可视化Image数据
    image_type: 'smiles' or 'scaffold'
    """
    if idx >= len(image_data):
        return None
    
    img_data = image_data[idx]
    
    # 检查数据格式
    if isinstance(img_data, dict):
        # 如果是字典，尝试获取图像数据
        if image_type == 'smiles' and 'smiles_image' in img_data:
            img_array = img_data['smiles_image']
        elif image_type == 'scaffold' and 'scaffold_image' in img_data:
            img_array = img_data['scaffold_image']
        elif 'image' in img_data:
            img_array = img_data['image']
        elif 'data' in img_data:
            img_array = img_data['data']
        else:
            # 返回None如果找不到图像数据
            return None
    else:
        img_array = img_data
    
    # 如果是tensor，转换为numpy
    if torch.is_tensor(img_array):
        img_array = img_array.numpy()
    
    # 确保是numpy数组
    if not isinstance(img_array, np.ndarray):
        return None
    
    # 如果是CHW格式，转换为HWC
    if len(img_array.shape) == 3 and img_array.shape[0] == 3:
        img_array = np.transpose(img_array, (1, 2, 0))
    
    return img_array

def create_visualization_report():
    """创建完整的可视化报告"""
    print("\n" + "="*70)
    print("🔍 多模态数据可视化分析")
    print("="*70)
    
    # 路径设置
    data_dir = Path('/root/text2Mol/scaffold-mol-generation/Datasets')
    output_dir = Path('/root/text2Mol/scaffold-mol-generation/visualization_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载数据
    print("\n📊 加载数据...")
    
    # CSV数据
    csv_path = data_dir / 'test_small_with_scaffold.csv'
    df = pd.read_csv(csv_path)
    print(f"  CSV数据: {len(df)} 样本")
    
    # Graph数据
    graph_path = data_dir / 'graph' / 'test_small_graphs.pkl'
    graph_data = load_pkl_data(graph_path)
    print(f"  Graph数据: {len(graph_data)} 样本")
    
    # Image数据
    image_path = data_dir / 'image' / 'test_small_images.pkl'
    image_data = load_pkl_data(image_path)
    print(f"  Image数据: {len(image_data)} 样本")
    
    # 2. 展示样本数据
    print("\n📋 数据样本展示:")
    print("-"*50)
    
    # 选择前5个样本进行展示
    num_samples = min(5, len(df))
    
    # 创建HTML报告
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>多模态分子数据可视化</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
            h2 { color: #666; margin-top: 30px; }
            .sample { 
                background: white; 
                padding: 20px; 
                margin: 20px 0; 
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .modality { 
                display: inline-block; 
                margin: 10px; 
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                vertical-align: top;
            }
            .smiles-box {
                background: #f9f9f9;
                padding: 10px;
                border-left: 4px solid #4CAF50;
                margin: 10px 0;
                font-family: monospace;
            }
            .graph-info {
                background: #e8f4f8;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
            }
            img { max-width: 299px; height: auto; border: 1px solid #ddd; }
            table { border-collapse: collapse; width: 100%; margin: 10px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
            .stats { background: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0; }
        </style>
    </head>
    <body>
        <h1>🧬 多模态分子数据可视化报告</h1>
        <div class="stats">
            <h3>📊 数据集统计</h3>
            <ul>
                <li><strong>总样本数:</strong> """ + str(len(df)) + """ 个</li>
                <li><strong>数据来源:</strong> ChEBI-20 测试集</li>
                <li><strong>生成时间:</strong> """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</li>
            </ul>
        </div>
    """
    
    # 为每个样本创建可视化
    for i in range(num_samples):
        row = df.iloc[i]
        
        print(f"\n样本 {i+1}:")
        print(f"  CID: {row['CID']}")
        print(f"  Text: {row['text'][:50]}...")
        print(f"  SMILES: {row['SMILES']}")
        print(f"  Scaffold: {row['scaffold']}")
        
        # 创建分子图像
        mol = Chem.MolFromSmiles(row['SMILES'])
        scaffold_mol = Chem.MolFromSmiles(row['scaffold'])
        
        if mol and scaffold_mol:
            # 生成分子图像
            mol_img = Draw.MolToImage(mol, size=(299, 299))
            scaffold_img = Draw.MolToImage(scaffold_mol, size=(299, 299))
            
            # 保存图像
            mol_img_path = output_dir / f'mol_{i}.png'
            scaffold_img_path = output_dir / f'scaffold_{i}.png'
            mol_img.save(mol_img_path)
            scaffold_img.save(scaffold_img_path)
            
            # Graph信息
            graph_info = visualize_graph_data(graph_data, i)
            
            # Image数据 - 获取SMILES图像和Scaffold图像
            smiles_img_array = visualize_image_data(image_data, i, 'smiles')
            scaffold_img_array = visualize_image_data(image_data, i, 'scaffold')
            
            if smiles_img_array is not None:
                # 保存SMILES图像
                if smiles_img_array.dtype != np.uint8:
                    smiles_img_array = (smiles_img_array * 255).astype(np.uint8)
                img = PILImage.fromarray(smiles_img_array)
                img_path = output_dir / f'processed_smiles_img_{i}.png'
                img.save(img_path)
            
            if scaffold_img_array is not None:
                # 保存Scaffold图像
                if scaffold_img_array.dtype != np.uint8:
                    scaffold_img_array = (scaffold_img_array * 255).astype(np.uint8)
                img = PILImage.fromarray(scaffold_img_array)
                scaffold_proc_path = output_dir / f'processed_scaffold_img_{i}.png'
                img.save(scaffold_proc_path)
            
            # 添加到HTML
            html_content += f"""
            <div class="sample">
                <h2>样本 {i+1} (CID: {row['CID']})</h2>
                
                <h3>📝 文本描述</h3>
                <p>{row['text']}</p>
                
                <h3>🧪 SMILES表示</h3>
                <div class="smiles-box">
                    <strong>完整分子:</strong> {row['SMILES']}<br>
                    <strong>Scaffold:</strong> {row['scaffold']}
                </div>
                
                <h3>🖼️ 分子图像</h3>
                <div>
                    <div class="modality">
                        <h4>完整分子</h4>
                        <img src="mol_{i}.png" alt="Molecule">
                    </div>
                    <div class="modality">
                        <h4>Scaffold</h4>
                        <img src="scaffold_{i}.png" alt="Scaffold">
                    </div>
                    <div class="modality">
                        <h4>处理后的SMILES图像</h4>
                        <img src="processed_smiles_img_{i}.png" alt="Processed SMILES">
                    </div>
                    <div class="modality">
                        <h4>处理后的Scaffold图像</h4>
                        <img src="processed_scaffold_img_{i}.png" alt="Processed Scaffold">
                    </div>
                </div>
                
                <h3>📊 Graph数据信息</h3>
                <div class="graph-info">
                    <table>
                        <tr><th>属性</th><th>值</th></tr>
                        <tr><td>原子数</td><td>{graph_info['num_atoms'] if graph_info else 'N/A'}</td></tr>
                        <tr><td>化学键数</td><td>{graph_info['num_bonds'] if graph_info else 'N/A'}</td></tr>
                        <tr><td>原子特征维度</td><td>{graph_info['atom_features_dim'] if graph_info else 'N/A'}</td></tr>
                        <tr><td>包含边特征</td><td>{'是' if graph_info and graph_info['has_edge_attr'] else '否'}</td></tr>
                    </table>
                </div>
            </div>
            """
    
    html_content += """
        <h2>📈 模态转换示例</h2>
        <div class="stats">
            <p>系统支持以下9种输入-输出组合：</p>
            <table>
                <tr><th>输入模态</th><th>输出模态</th><th>状态</th></tr>
                <tr><td>SMILES + Text</td><td>SMILES</td><td>✅ 已实现</td></tr>
                <tr><td>SMILES + Text</td><td>Graph</td><td>✅ 已实现</td></tr>
                <tr><td>SMILES + Text</td><td>Image</td><td>✅ 已实现</td></tr>
                <tr><td>Graph + Text</td><td>SMILES</td><td>✅ 已实现</td></tr>
                <tr><td>Graph + Text</td><td>Graph</td><td>✅ 已实现</td></tr>
                <tr><td>Graph + Text</td><td>Image</td><td>✅ 已实现</td></tr>
                <tr><td>Image + Text</td><td>SMILES</td><td>✅ 已实现</td></tr>
                <tr><td>Image + Text</td><td>Graph</td><td>✅ 已实现</td></tr>
                <tr><td>Image + Text</td><td>Image</td><td>✅ 已实现</td></tr>
            </table>
        </div>
    </body>
    </html>
    """
    
    # 保存HTML报告
    html_path = output_dir / 'visualization_report.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("\n" + "="*70)
    print("✅ 可视化报告生成完成！")
    print(f"📁 输出目录: {output_dir}")
    print(f"🌐 HTML报告: {html_path}")
    print(f"🖼️ 包含 {num_samples} 个样本的完整可视化")
    print("="*70)
    
    # 生成数据统计
    print("\n📊 数据集详细统计:")
    print(f"  - 测试样本总数: {len(df)}")
    print(f"  - Graph数据完整性: {len(graph_data)}/{len(df)} ({100*len(graph_data)/len(df):.1f}%)")
    print(f"  - Image数据完整性: {len(image_data)}/{len(df)} ({100*len(image_data)/len(df):.1f}%)")
    
    # 检查SMILES有效性
    valid_smiles = 0
    valid_scaffolds = 0
    for _, row in df.iterrows():
        try:
            if pd.notna(row['SMILES']) and Chem.MolFromSmiles(row['SMILES']) is not None:
                valid_smiles += 1
        except:
            pass
        try:
            if pd.notna(row['scaffold']) and Chem.MolFromSmiles(row['scaffold']) is not None:
                valid_scaffolds += 1
        except:
            pass
    
    print(f"  - 有效SMILES: {valid_smiles}/{len(df)} ({100*valid_smiles/len(df):.1f}%)")
    print(f"  - 有效Scaffold: {valid_scaffolds}/{len(df)} ({100*valid_scaffolds/len(df):.1f}%)")

if __name__ == "__main__":
    create_visualization_report()