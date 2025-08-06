"""
多模态数据预处理器
将SMILES转换为Graph和Image格式，支持7种输入输出组合
"""

import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from tqdm import tqdm
import os
import pickle
import json

# 分子处理
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

# 图处理
from torch_geometric.data import Data
import torch_geometric.transforms as T

# 图像处理
from PIL import Image
import matplotlib.pyplot as plt

# 工具
from ..utils.mol_utils import MolecularUtils
from ..utils.scaffold_utils import ScaffoldExtractor

logger = logging.getLogger(__name__)

class MultiModalPreprocessor:
    """
    多模态数据预处理器
    支持SMILES → Graph, SMILES → Image转换
    """
    
    def __init__(self,
                 image_size: Tuple[int, int] = (224, 224),
                 bond_features: bool = True,
                 cache_dir: str = "cache/multimodal"):
        """
        Args:
            image_size: 分子图像尺寸
            bond_features: 是否包含边特征
            cache_dir: 缓存目录
        """
        self.image_size = image_size
        self.bond_features = bond_features
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化工具
        self.mol_utils = MolecularUtils()
        self.scaffold_extractor = ScaffoldExtractor()
        
        # 原子特征映射
        self.atom_features = {
            'atomic_num': list(range(1, 119)),  # 1-118号元素
            'degree': [0, 1, 2, 3, 4, 5, 6],
            'formal_charge': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
            'hybridization': [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
                Chem.rdchem.HybridizationType.UNSPECIFIED,
            ],
            'aromatic': [False, True],
            'in_ring': [False, True]
        }
        
        # 键特征映射
        self.bond_features_map = {
            'bond_type': [
                Chem.rdchem.BondType.SINGLE,
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE,
                Chem.rdchem.BondType.AROMATIC,
            ],
            'conjugated': [False, True],
            'in_ring': [False, True]
        }
        
        logger.info("多模态预处理器初始化完成")
    
    def smiles_to_graph(self, smiles: str) -> Optional[Data]:
        """
        将SMILES转换为PyTorch Geometric图数据
        
        Args:
            smiles: SMILES字符串
            
        Returns:
            torch_geometric.data.Data对象，包含节点和边特征
        """
        try:
            # 创建RDKit分子对象
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"无法解析SMILES: {smiles}")
                return None
            
            # 不添加氢原子，保持与GINEncoder一致
            # mol = Chem.AddHs(mol)
            
            # 节点特征 (原子特征)
            node_features = []
            for atom in mol.GetAtoms():
                features = self._get_atom_features(atom)
                node_features.append(features)
            
            # 转换为tensor
            x = torch.tensor(node_features, dtype=torch.float)
            
            # 边特征
            edge_indices = []
            edge_features = []
            
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                # 无向图，添加双向边
                edge_indices.extend([[i, j], [j, i]])
                
                # GINEncoder不使用边特征，跳过
                # if self.bond_features:
                #     bond_feat = self._get_bond_features(bond)
                #     edge_features.extend([bond_feat, bond_feat])
                # else:
                #     edge_features.extend([[], []])
            
            # 处理孤立节点
            if len(edge_indices) == 0:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            
            # 创建图数据对象 - 不包含边特征以保持与GINEncoder兼容
            data = Data(
                x=x,
                edge_index=edge_index
                # 不包含edge_attr，因为GINEncoder不使用边特征
            )
            
            return data
            
        except Exception as e:
            logger.error(f"SMILES转Graph失败 {smiles}: {e}")
            return None
    
    def smiles_to_image(self, smiles: str) -> Optional[np.ndarray]:
        """
        将SMILES转换为分子图像
        
        Args:
            smiles: SMILES字符串
            
        Returns:
            numpy数组格式的RGB图像 (H, W, 3)
        """
        try:
            # 创建分子对象
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"无法解析SMILES: {smiles}")
                return None
            
            # 生成2D坐标
            AllChem.Compute2DCoords(mol)
            
            # 绘制分子图像
            img = Draw.MolToImage(
                mol, 
                size=self.image_size,
                kekulize=True,
                wedgeBonds=True,
                imageType='png',
                fitImage=True,
                options=None
            )
            
            # 转换为numpy数组
            img_array = np.array(img)
            
            # 确保是RGB格式
            if len(img_array.shape) == 2:  # 灰度图
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[-1] == 4:  # RGBA
                img_array = img_array[:, :, :3]
            
            # 转换为tensor
            import torch
            img_tensor = torch.from_numpy(img_array).float()
            # 调整维度顺序为 (C, H, W)
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.permute(2, 0, 1)
            return img_tensor
            
        except Exception as e:
            logger.error(f"SMILES转Image失败 {smiles}: {e}")
            return None
    
    def _get_atom_features(self, atom) -> List[float]:
        """提取原子特征 - 9维简单数值特征，与GINEncoder兼容"""
        features = [
            float(atom.GetAtomicNum()),           # 原子序数
            float(atom.GetDegree()),              # 度数
            float(atom.GetFormalCharge()),        # 形式电荷
            float(atom.GetHybridization()),       # 杂化类型
            float(atom.GetIsAromatic()),          # 芳香性
            float(atom.GetTotalNumHs()),          # 氢原子数
            float(atom.GetNumRadicalElectrons()), # 自由基电子数
            float(atom.IsInRing()),               # 是否在环中
            float(atom.GetChiralTag())            # 手性标记
        ]
        return features
    
    def _get_bond_features(self, bond) -> List[float]:
        """提取键特征"""
        features = []
        
        # 键类型
        bond_type = bond.GetBondType()
        features.extend(self._one_hot_encode(bond_type, self.bond_features_map['bond_type']))
        
        # 共轭
        conjugated = bond.GetIsConjugated()
        features.extend(self._one_hot_encode(conjugated, self.bond_features_map['conjugated']))
        
        # 是否在环中
        in_ring = bond.IsInRing()
        features.extend(self._one_hot_encode(in_ring, self.bond_features_map['in_ring']))
        
        return features
    
    def _one_hot_encode(self, value, allowable_set) -> List[float]:
        """One-hot编码"""
        if value in allowable_set:
            return [1.0 if value == s else 0.0 for s in allowable_set]
        else:
            # 未知值用全0表示
            return [0.0] * len(allowable_set)
    
    def process_dataset(self, 
                       csv_path: str,
                       output_dir: str,
                       batch_size: int = 1000) -> Dict[str, Any]:
        """
        处理整个数据集，生成多模态数据
        
        Args:
            csv_path: 输入CSV文件路径
            output_dir: 输出目录
            batch_size: 批处理大小
            
        Returns:
            处理统计信息
        """
        logger.info(f"开始处理数据集: {csv_path}")
        
        # 读取数据
        df = pd.read_csv(csv_path)
        logger.info(f"加载数据: {len(df)} 条记录")
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化结果存储
        results = {
            'scaffold_graphs': [],
            'scaffold_images': [],
            'target_graphs': [],
            'target_images': [],
            'metadata': [],
            'statistics': {
                'total_samples': len(df),
                'successful_scaffold_graphs': 0,
                'successful_scaffold_images': 0, 
                'successful_target_graphs': 0,
                'successful_target_images': 0,
                'failed_samples': []
            }
        }
        
        # 提取scaffold（如果不存在）
        if 'scaffold' not in df.columns:
            logger.info("提取分子scaffold...")
            scaffolds = []
            for smiles in tqdm(df['SMILES'], desc="提取scaffold"):
                scaffold = self.scaffold_extractor.get_murcko_scaffold(smiles)
                scaffolds.append(scaffold if scaffold else smiles)
            df['scaffold'] = scaffolds
        
        # 重命名列以保持一致
        if 'description' in df.columns and 'text' not in df.columns:
            df['text'] = df['description']
        
        # 批处理数据
        num_batches = (len(df) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            
            logger.info(f"处理批次 {batch_idx + 1}/{num_batches} ({start_idx}-{end_idx})")
            
            batch_results = self._process_batch(batch_df, start_idx)
            
            # 合并结果
            for key in ['scaffold_graphs', 'scaffold_images', 'target_graphs', 'target_images', 'metadata']:
                results[key].extend(batch_results[key])
            
            # 更新统计信息
            for key in ['successful_scaffold_graphs', 'successful_scaffold_images', 
                       'successful_target_graphs', 'successful_target_images']:
                results['statistics'][key] += batch_results['statistics'][key]
            results['statistics']['failed_samples'].extend(batch_results['statistics']['failed_samples'])
        
        # 保存结果
        self._save_processed_data(results, output_path)
        
        # 保存更新的CSV
        df.to_csv(output_path / 'processed_dataset.csv', index=False)
        
        logger.info("数据集处理完成")
        logger.info(f"统计信息: {results['statistics']}")
        
        return results['statistics']
    
    def _process_batch(self, batch_df: pd.DataFrame, start_idx: int) -> Dict[str, Any]:
        """处理一个批次的数据"""
        batch_results = {
            'scaffold_graphs': [],
            'scaffold_images': [],
            'target_graphs': [],
            'target_images': [],
            'metadata': [],
            'statistics': {
                'successful_scaffold_graphs': 0,
                'successful_scaffold_images': 0,
                'successful_target_graphs': 0,
                'successful_target_images': 0,
                'failed_samples': []
            }
        }
        
        for idx, row in batch_df.iterrows():
            try:
                sample_metadata = {
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
                    scaffold_graph = self.smiles_to_graph(scaffold_smiles)
                    if scaffold_graph is not None:
                        batch_results['scaffold_graphs'].append(scaffold_graph)
                        batch_results['statistics']['successful_scaffold_graphs'] += 1
                    else:
                        batch_results['scaffold_graphs'].append(None)
                    
                    # Scaffold → Image
                    scaffold_image = self.smiles_to_image(scaffold_smiles)
                    if scaffold_image is not None:
                        batch_results['scaffold_images'].append(scaffold_image)
                        batch_results['statistics']['successful_scaffold_images'] += 1
                    else:
                        batch_results['scaffold_images'].append(None)
                else:
                    batch_results['scaffold_graphs'].append(None)
                    batch_results['scaffold_images'].append(None)
                
                # 处理target
                target_smiles = row.get('SMILES', '')
                if target_smiles:
                    # Target → Graph
                    target_graph = self.smiles_to_graph(target_smiles)
                    if target_graph is not None:
                        batch_results['target_graphs'].append(target_graph)
                        batch_results['statistics']['successful_target_graphs'] += 1
                    else:
                        batch_results['target_graphs'].append(None)
                    
                    # Target → Image
                    target_image = self.smiles_to_image(target_smiles)
                    if target_image is not None:
                        batch_results['target_images'].append(target_image)
                        batch_results['statistics']['successful_target_images'] += 1
                    else:
                        batch_results['target_images'].append(None)
                else:
                    batch_results['target_graphs'].append(None)
                    batch_results['target_images'].append(None)
                
                batch_results['metadata'].append(sample_metadata)
                
            except Exception as e:
                logger.error(f"处理样本失败 {idx}: {e}")
                batch_results['statistics']['failed_samples'].append(idx)
                # 添加空数据保持索引一致
                batch_results['scaffold_graphs'].append(None)
                batch_results['scaffold_images'].append(None)
                batch_results['target_graphs'].append(None)
                batch_results['target_images'].append(None)
                batch_results['metadata'].append({'original_index': idx, 'error': str(e)})
        
        return batch_results
    
    def _save_processed_data(self, results: Dict[str, Any], output_path: Path):
        """保存处理后的数据"""
        logger.info("保存处理后的数据...")
        
        # 保存图数据
        graphs_dir = output_path / 'graphs'
        graphs_dir.mkdir(exist_ok=True)
        
        torch.save(results['scaffold_graphs'], graphs_dir / 'scaffold_graphs.pt')
        torch.save(results['target_graphs'], graphs_dir / 'target_graphs.pt')
        
        # 保存图像数据 (使用pickle处理不同形状的数组)
        images_dir = output_path / 'images'
        images_dir.mkdir(exist_ok=True)
        
        # 使用pickle保存以避免形状不一致的问题
        import pickle
        with open(images_dir / 'scaffold_images.pkl', 'wb') as f:
            pickle.dump(results['scaffold_images'], f)
        with open(images_dir / 'target_images.pkl', 'wb') as f:
            pickle.dump(results['target_images'], f)
        
        # 保存元数据
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(results['metadata'], f, indent=2)
        
        # 保存统计信息
        with open(output_path / 'statistics.json', 'w') as f:
            json.dump(results['statistics'], f, indent=2)
        
        logger.info(f"数据保存完成: {output_path}")
    
    def load_processed_data(self, data_dir: str) -> Dict[str, Any]:
        """加载处理后的数据"""
        data_path = Path(data_dir)
        
        logger.info(f"加载处理后的数据: {data_path}")
        
        # 加载图数据 (使用weights_only=False处理PyTorch安全警告)
        scaffold_graphs = torch.load(data_path / 'graphs' / 'scaffold_graphs.pt', weights_only=False)
        target_graphs = torch.load(data_path / 'graphs' / 'target_graphs.pt', weights_only=False)
        
        # 加载图像数据
        import pickle
        with open(data_path / 'images' / 'scaffold_images.pkl', 'rb') as f:
            scaffold_images = pickle.load(f)
        with open(data_path / 'images' / 'target_images.pkl', 'rb') as f:
            target_images = pickle.load(f)
        
        # 加载元数据
        with open(data_path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # 加载统计信息
        with open(data_path / 'statistics.json', 'r') as f:
            statistics = json.load(f)
        
        return {
            'scaffold_graphs': scaffold_graphs,
            'scaffold_images': scaffold_images,
            'target_graphs': target_graphs,
            'target_images': target_images,
            'metadata': metadata,
            'statistics': statistics
        }


def test_preprocessor():
    """测试多模态预处理器"""
    logger.info("测试多模态预处理器...")
    
    # 初始化预处理器
    preprocessor = MultiModalPreprocessor()
    
    # 测试SMILES
    test_smiles = "CCO"  # 乙醇
    
    # 测试SMILES → Graph
    graph = preprocessor.smiles_to_graph(test_smiles)
    if graph is not None:
        logger.info(f"✅ SMILES → Graph 成功: {graph}")
        logger.info(f"   节点数: {graph.x.shape[0]}, 边数: {graph.edge_index.shape[1]}")
    else:
        logger.error("❌ SMILES → Graph 失败")
    
    # 测试SMILES → Image
    image = preprocessor.smiles_to_image(test_smiles)
    if image is not None:
        logger.info(f"✅ SMILES → Image 成功: {image.shape}")
    else:
        logger.error("❌ SMILES → Image 失败")
    
    logger.info("多模态预处理器测试完成")


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    test_preprocessor()