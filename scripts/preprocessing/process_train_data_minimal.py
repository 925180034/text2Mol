#!/usr/bin/env python3
"""
最小化训练数据预处理脚本
处理大量数据时减少内存占用
"""

import sys
import logging
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gc
import pickle

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from scaffold_mol_gen.data.multimodal_preprocessor import MultiModalPreprocessor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_train_data_in_chunks():
    """分块处理训练数据以避免内存问题"""
    
    # 输入输出路径
    train_csv = "Datasets/train.csv"
    output_dir = "/root/autodl-tmp/multimodal-data/train"
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir + "/graphs").mkdir(exist_ok=True) 
    Path(output_dir + "/images").mkdir(exist_ok=True)
    
    # 初始化预处理器
    preprocessor = MultiModalPreprocessor()
    
    # 读取数据
    logger.info("读取训练数据...")
    df = pd.read_csv(train_csv)
    total_samples = len(df)
    logger.info(f"总样本数: {total_samples}")
    
    # 如果已存在，先删除
    if Path(output_dir + "/processed_dataset.csv").exists():
        logger.info("删除已存在的处理文件...")
        Path(output_dir + "/processed_dataset.csv").unlink()
    
    # 分块处理参数
    chunk_size = 1000
    num_chunks = (total_samples + chunk_size - 1) // chunk_size
    
    # 结果存储
    all_scaffold_graphs = []
    all_scaffold_images = []
    all_target_graphs = []
    all_target_images = []
    all_metadata = []
    
    statistics = {
        'total_samples': total_samples,
        'successful_scaffold_graphs': 0,
        'successful_scaffold_images': 0,
        'successful_target_graphs': 0,
        'successful_target_images': 0,
        'failed_samples': []
    }
    
    # 处理每个块
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_samples)
        
        logger.info(f"处理块 {chunk_idx + 1}/{num_chunks} ({start_idx}-{end_idx})")
        
        # 获取当前块
        chunk_df = df.iloc[start_idx:end_idx].copy()
        
        # 添加scaffold列（如果不存在）
        if 'scaffold' not in chunk_df.columns:
            scaffolds = []
            for smiles in tqdm(chunk_df['SMILES'], desc="提取scaffold"):
                scaffold = preprocessor.scaffold_extractor.get_murcko_scaffold(smiles)
                scaffolds.append(scaffold if scaffold else smiles)
            chunk_df['scaffold'] = scaffolds
        
        # 重命名列
        if 'description' in chunk_df.columns and 'text' not in chunk_df.columns:
            chunk_df['text'] = chunk_df['description']
        
        # 处理当前块
        chunk_scaffold_graphs = []
        chunk_scaffold_images = []
        chunk_target_graphs = []
        chunk_target_images = []
        chunk_metadata = []
        
        for idx, row in tqdm(chunk_df.iterrows(), total=len(chunk_df), desc="处理样本"):
            try:
                # 元数据
                metadata = {
                    'original_index': idx,
                    'cid': row.get('CID', ''),
                    'text': row.get('text', ''),
                    'scaffold_smiles': row.get('scaffold', ''),
                    'target_smiles': row.get('SMILES', '')
                }
                
                # 处理scaffold
                scaffold_smiles = row.get('scaffold', '')
                if scaffold_smiles:
                    # Scaffold → Graph
                    scaffold_graph = preprocessor.smiles_to_graph(scaffold_smiles)
                    if scaffold_graph is not None:
                        chunk_scaffold_graphs.append(scaffold_graph)
                        statistics['successful_scaffold_graphs'] += 1
                    else:
                        chunk_scaffold_graphs.append(None)
                    
                    # Scaffold → Image
                    scaffold_image = preprocessor.smiles_to_image(scaffold_smiles)
                    if scaffold_image is not None:
                        chunk_scaffold_images.append(scaffold_image)
                        statistics['successful_scaffold_images'] += 1
                    else:
                        chunk_scaffold_images.append(None)
                else:
                    chunk_scaffold_graphs.append(None)
                    chunk_scaffold_images.append(None)
                
                # 处理target
                target_smiles = row.get('SMILES', '')
                if target_smiles:
                    # Target → Graph
                    target_graph = preprocessor.smiles_to_graph(target_smiles)
                    if target_graph is not None:
                        chunk_target_graphs.append(target_graph)
                        statistics['successful_target_graphs'] += 1
                    else:
                        chunk_target_graphs.append(None)
                    
                    # Target → Image
                    target_image = preprocessor.smiles_to_image(target_smiles)
                    if target_image is not None:
                        chunk_target_images.append(target_image)
                        statistics['successful_target_images'] += 1
                    else:
                        chunk_target_images.append(None)
                else:
                    chunk_target_graphs.append(None)
                    chunk_target_images.append(None)
                
                chunk_metadata.append(metadata)
                
            except Exception as e:
                logger.error(f"处理样本失败 {idx}: {e}")
                statistics['failed_samples'].append(idx)
                # 添加空数据保持索引一致
                chunk_scaffold_graphs.append(None)
                chunk_scaffold_images.append(None)
                chunk_target_graphs.append(None)
                chunk_target_images.append(None)
                chunk_metadata.append({'original_index': idx, 'error': str(e)})
        
        # 合并到总结果
        all_scaffold_graphs.extend(chunk_scaffold_graphs)
        all_scaffold_images.extend(chunk_scaffold_images)
        all_target_graphs.extend(chunk_target_graphs)
        all_target_images.extend(chunk_target_images)
        all_metadata.extend(chunk_metadata)
        
        # 清理内存
        del chunk_scaffold_graphs, chunk_scaffold_images
        del chunk_target_graphs, chunk_target_images
        del chunk_metadata, chunk_df
        gc.collect()
        
        # 每5个块保存一次中间结果
        if (chunk_idx + 1) % 5 == 0:
            logger.info(f"保存中间结果... (已处理 {chunk_idx + 1}/{num_chunks} 块)")
            save_intermediate_results(
                output_dir, all_scaffold_graphs, all_scaffold_images,
                all_target_graphs, all_target_images, all_metadata, 
                statistics, chunk_idx + 1
            )
    
    # 保存最终结果
    logger.info("保存最终结果...")
    save_final_results(
        output_dir, all_scaffold_graphs, all_scaffold_images,
        all_target_graphs, all_target_images, all_metadata, 
        statistics, df
    )
    
    logger.info("训练数据处理完成!")
    logger.info(f"统计信息: {statistics}")
    
    return statistics

def save_intermediate_results(output_dir, scaffold_graphs, scaffold_images, 
                            target_graphs, target_images, metadata, 
                            statistics, chunk_num):
    """保存中间结果"""
    backup_dir = Path(output_dir) / f"backup_chunk_{chunk_num}"
    backup_dir.mkdir(exist_ok=True)
    
    # 保存图数据
    torch.save(scaffold_graphs, backup_dir / 'scaffold_graphs.pt')
    torch.save(target_graphs, backup_dir / 'target_graphs.pt')
    
    # 保存图像数据
    with open(backup_dir / 'scaffold_images.pkl', 'wb') as f:
        pickle.dump(scaffold_images, f)
    with open(backup_dir / 'target_images.pkl', 'wb') as f:
        pickle.dump(target_images, f)
    
    # 保存元数据
    import json
    with open(backup_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f)
    with open(backup_dir / 'statistics.json', 'w') as f:
        json.dump(statistics, f)

def save_final_results(output_dir, scaffold_graphs, scaffold_images,
                      target_graphs, target_images, metadata, 
                      statistics, df):
    """保存最终结果"""
    output_path = Path(output_dir)
    
    # 保存图数据
    graphs_dir = output_path / 'graphs'
    graphs_dir.mkdir(exist_ok=True)
    torch.save(scaffold_graphs, graphs_dir / 'scaffold_graphs.pt')
    torch.save(target_graphs, graphs_dir / 'target_graphs.pt')
    logger.info("✅ 图数据保存完成")
    
    # 保存图像数据
    images_dir = output_path / 'images'
    images_dir.mkdir(exist_ok=True)
    with open(images_dir / 'scaffold_images.pkl', 'wb') as f:
        pickle.dump(scaffold_images, f)
    with open(images_dir / 'target_images.pkl', 'wb') as f:
        pickle.dump(target_images, f)
    logger.info("✅ 图像数据保存完成")
    
    # 保存更新的CSV
    df.to_csv(output_path / 'processed_dataset.csv', index=False)
    logger.info("✅ 处理后的CSV保存完成")
    
    # 保存元数据和统计
    import json
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    with open(output_path / 'statistics.json', 'w') as f:
        json.dump(statistics, f, indent=2)
    logger.info("✅ 元数据和统计信息保存完成")

if __name__ == "__main__":
    # 运行处理
    stats = process_train_data_in_chunks()
    
    print("\n=== 处理完成 ===")
    print(f"总样本数: {stats['total_samples']}")
    print(f"成功的Scaffold图: {stats['successful_scaffold_graphs']}")
    print(f"成功的Scaffold图像: {stats['successful_scaffold_images']}")
    print(f"成功的Target图: {stats['successful_target_graphs']}")
    print(f"成功的Target图像: {stats['successful_target_images']}")
    print(f"失败样本数: {len(stats['failed_samples'])}")