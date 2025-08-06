"""
多模态分子生成数据集
适配端到端模型的数据加载器
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
import random
from rdkit import Chem
from rdkit.Chem import Descriptors

logger = logging.getLogger(__name__)


class MultiModalMolecularDataset(Dataset):
    """
    多模态分子生成数据集
    支持Scaffold + Text → SMILES的训练
    """
    
    def __init__(self,
                 csv_path: str,
                 scaffold_modality: str = 'smiles',
                 max_text_length: int = 128,
                 max_smiles_length: int = 128,
                 filter_invalid: bool = True,
                 augment_data: bool = False,
                 scaffold_column: str = 'scaffold',
                 text_column: str = 'text',
                 smiles_column: str = 'SMILES'):
        """
        Args:
            csv_path: CSV文件路径
            scaffold_modality: Scaffold模态 ('smiles', 'graph', 'image')
            max_text_length: 最大文本长度
            max_smiles_length: 最大SMILES长度
            filter_invalid: 是否过滤无效分子
            augment_data: 是否进行数据增强
            scaffold_column: Scaffold列名
            text_column: 文本列名
            smiles_column: SMILES列名
        """
        self.csv_path = csv_path
        self.scaffold_modality = scaffold_modality
        self.max_text_length = max_text_length
        self.max_smiles_length = max_smiles_length
        self.filter_invalid = filter_invalid
        self.augment_data = augment_data
        
        # 加载数据
        logger.info(f"加载数据集: {csv_path}")
        self.data = pd.read_csv(csv_path)
        logger.info(f"原始数据量: {len(self.data)}")
        
        # 重命名列以保持一致性 - 处理实际CSV列名
        column_mapping = {}
        if scaffold_column in self.data.columns:
            column_mapping[scaffold_column] = 'scaffold'
        if text_column in self.data.columns:
            column_mapping[text_column] = 'text'
        elif 'description' in self.data.columns:
            column_mapping['description'] = 'text'  # ChEBI数据集使用description列
        if smiles_column in self.data.columns:
            column_mapping[smiles_column] = 'target_smiles'
        
        if column_mapping:
            self.data = self.data.rename(columns=column_mapping)
        
        # 数据清理和过滤
        self._clean_data()
        
        logger.info(f"清理后数据量: {len(self.data)}")
        logger.info(f"Scaffold模态: {scaffold_modality}")
        
    def _clean_data(self):
        """清理和过滤数据"""
        # 生成scaffold列（如果不存在）
        if 'scaffold' not in self.data.columns and 'target_smiles' in self.data.columns:
            logger.info("从SMILES提取scaffold...")
            from ..utils.scaffold_utils import ScaffoldExtractor
            scaffold_extractor = ScaffoldExtractor()
            scaffolds = []
            
            for smiles in self.data['target_smiles']:
                try:
                    scaffold = scaffold_extractor.extract_scaffold(str(smiles))
                    scaffolds.append(scaffold if scaffold else smiles)  # 回退到原始SMILES
                except:
                    scaffolds.append(str(smiles))  # 出错时使用原始SMILES
            
            self.data['scaffold'] = scaffolds
            logger.info(f"scaffold提取完成: {len([s for s in scaffolds if s])}")
        
        # 删除空值
        initial_size = len(self.data)
        required_cols = ['text', 'target_smiles']
        if 'scaffold' in self.data.columns:
            required_cols.append('scaffold')
        self.data = self.data.dropna(subset=required_cols)
        logger.info(f"删除空值: {initial_size} → {len(self.data)}")
        
        if self.filter_invalid:
            # 过滤无效的SMILES
            valid_mask = self.data['target_smiles'].apply(self._is_valid_smiles)
            self.data = self.data[valid_mask].reset_index(drop=True)
            logger.info(f"过滤无效SMILES后: {len(self.data)}")
            
            # 过滤无效的Scaffold（如果存在）
            if 'scaffold' in self.data.columns:
                valid_scaffold_mask = self.data['scaffold'].apply(self._is_valid_smiles)
                self.data = self.data[valid_scaffold_mask].reset_index(drop=True)
                logger.info(f"过滤无效Scaffold后: {len(self.data)}")
        
        # 长度过滤
        text_len_mask = self.data['text'].str.len() <= self.max_text_length * 4  # 粗略估计
        smiles_len_mask = self.data['target_smiles'].str.len() <= self.max_smiles_length
        
        self.data = self.data[text_len_mask & smiles_len_mask].reset_index(drop=True)
        logger.info(f"长度过滤后: {len(self.data)}")
    
    def _is_valid_smiles(self, smiles: str) -> bool:
        """检查SMILES是否有效"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个数据样本"""
        row = self.data.iloc[idx]
        
        # 基础数据
        scaffold = str(row['scaffold'])
        text = str(row['text'])
        target_smiles = str(row['target_smiles'])
        
        # 数据增强（可选）
        if self.augment_data and random.random() < 0.3:
            text = self._augment_text(text)
        
        # 根据scaffold_modality转换scaffold数据
        if self.scaffold_modality == 'graph':
            # 导入必要的转换函数
            from scaffold_mol_gen.data.multimodal_preprocessor import MultiModalPreprocessor
            preprocessor = MultiModalPreprocessor()
            # 将SMILES转换为Graph
            scaffold_data = preprocessor.smiles_to_graph(scaffold)
            if scaffold_data is None:
                # 如果转换失败，创建一个空图
                from torch_geometric.data import Data
                import torch
                scaffold_data = Data(
                    x=torch.zeros((1, 9)),
                    edge_index=torch.zeros((2, 0), dtype=torch.long)
                )
        elif self.scaffold_modality == 'image':
            # 将SMILES转换为Image
            from scaffold_mol_gen.data.multimodal_preprocessor import MultiModalPreprocessor
            preprocessor = MultiModalPreprocessor()
            scaffold_data = preprocessor.smiles_to_image(scaffold)
            if scaffold_data is None:
                # 如果转换失败，创建一个空图像
                import torch
                scaffold_data = torch.zeros((3, 224, 224))
            else:
                # 确保是tensor而不是numpy数组
                import torch
                if not isinstance(scaffold_data, torch.Tensor):
                    scaffold_data = torch.from_numpy(scaffold_data).float()
                else:
                    scaffold_data = scaffold_data.float()
                # 注意：不在这里移动到CUDA，让collate_batch处理
        else:
            # 默认为SMILES
            scaffold_data = scaffold
        
        sample = {
            'scaffold_data': scaffold_data,
            'text_data': text,
            'target_smiles': target_smiles,
            'scaffold_modality': self.scaffold_modality,
            'idx': idx
        }
        
        # 添加额外信息
        if 'CID' in row:
            sample['cid'] = row['CID']
        
        return sample
    
    def _augment_text(self, text: str) -> str:
        """简单的文本增强"""
        # 可以添加同义词替换、句子重排等
        return text
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        stats = {
            'total_samples': len(self.data),
            'avg_text_length': self.data['text'].str.len().mean(),
            'avg_smiles_length': self.data['target_smiles'].str.len().mean(),
            'avg_scaffold_length': self.data['scaffold'].str.len().mean(),
        }
        
        # 分子属性统计
        try:
            mol_weights = []
            logps = []
            
            for smiles in self.data['target_smiles'].head(1000):  # 采样1000个
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    mol_weights.append(Descriptors.MolWt(mol))
                    logps.append(Descriptors.MolLogP(mol))
            
            if mol_weights:
                stats.update({
                    'avg_molecular_weight': np.mean(mol_weights),
                    'avg_logp': np.mean(logps),
                })
        except Exception as e:
            logger.warning(f"分子属性统计失败: {e}")
        
        return stats


def create_data_loaders(train_csv: str,
                       val_csv: str,
                       test_csv: Optional[str] = None,
                       batch_size: int = 16,
                       num_workers: int = 4,
                       scaffold_modality: str = 'smiles',
                       **dataset_kwargs) -> Dict[str, DataLoader]:
    """
    创建训练、验证和测试数据加载器
    
    Args:
        train_csv: 训练数据CSV路径
        val_csv: 验证数据CSV路径  
        test_csv: 测试数据CSV路径（可选）
        batch_size: 批大小
        num_workers: 工作进程数
        scaffold_modality: Scaffold模态
        **dataset_kwargs: 数据集额外参数
        
    Returns:
        包含数据加载器的字典
    """
    logger.info("创建数据加载器...")
    
    # 创建数据集
    train_dataset = MultiModalMolecularDataset(
        train_csv, 
        scaffold_modality=scaffold_modality,
        **dataset_kwargs
    )
    
    # 为验证集设置特定参数
    val_kwargs = dataset_kwargs.copy()
    val_kwargs['filter_invalid'] = True  # 验证集总是过滤无效数据
    val_kwargs['augment_data'] = False   # 验证集不增强
    
    val_dataset = MultiModalMolecularDataset(
        val_csv,
        scaffold_modality=scaffold_modality,
        **val_kwargs
    )
    
    # 数据加载器配置
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),
        'collate_fn': collate_batch
    }
    
    # 创建数据加载器
    data_loaders = {
        'train': DataLoader(
            train_dataset,
            shuffle=True,
            **loader_kwargs
        ),
        'val': DataLoader(
            val_dataset,
            shuffle=False,
            **loader_kwargs
        )
    }
    
    # 测试集（可选）
    if test_csv and Path(test_csv).exists():
        # 为测试集设置特定参数
        test_kwargs = dataset_kwargs.copy()
        test_kwargs['filter_invalid'] = True
        test_kwargs['augment_data'] = False
        
        test_dataset = MultiModalMolecularDataset(
            test_csv,
            scaffold_modality=scaffold_modality,
            **test_kwargs
        )
        data_loaders['test'] = DataLoader(
            test_dataset,
            shuffle=False,
            **loader_kwargs
        )
    
    # 打印统计信息
    logger.info("数据集统计:")
    logger.info(f"  训练集: {len(train_dataset)} 样本")
    logger.info(f"  验证集: {len(val_dataset)} 样本")
    if 'test' in data_loaders:
        logger.info(f"  测试集: {len(data_loaders['test'].dataset)} 样本")
    
    # 打印训练集详细统计
    train_stats = train_dataset.get_statistics()
    logger.info("训练集统计:")
    for key, value in train_stats.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.2f}")
        else:
            logger.info(f"  {key}: {value}")
    
    return data_loaders


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    批处理整理函数
    
    Args:
        batch: 批数据列表
        
    Returns:
        整理后的批数据
    """
    # 提取各字段
    scaffold_data = [item['scaffold_data'] for item in batch]
    text_data = [item['text_data'] for item in batch] 
    target_smiles = [item['target_smiles'] for item in batch]
    scaffold_modality = batch[0]['scaffold_modality']  # 假设批内模态一致
    
    # 对于Graph模态，scaffold_data已经是Data对象列表，保持原样
    # 对于Image模态，scaffold_data是tensor列表，需要stack
    if scaffold_modality == 'image':
        import torch
        # 确保所有tensor都是FloatTensor并且stack
        tensor_list = []
        for data in scaffold_data:
            if isinstance(data, torch.Tensor):
                # 确保在CPU上且为float类型
                tensor_list.append(data.cpu().float())
            else:
                # 如果不是tensor，转换它
                tensor_list.append(torch.from_numpy(data).float())
        scaffold_data = torch.stack(tensor_list)
    # 对于Graph和SMILES模态，保持列表形式
    
    # 额外信息
    indices = [item['idx'] for item in batch]
    cids = [item.get('cid', None) for item in batch]
    
    collated = {
        'scaffold_data': scaffold_data,
        'text_data': text_data,
        'target_smiles': target_smiles,
        'scaffold_modality': scaffold_modality,
        'indices': indices,
        'batch_size': len(batch)
    }
    
    # 添加CID（如果存在）
    if any(cid is not None for cid in cids):
        collated['cids'] = cids
    
    return collated


def test_dataset():
    """测试数据集加载"""
    import os
    
    # 测试文件路径
    train_csv = "Datasets/train.csv"
    val_csv = "Datasets/validation.csv"
    
    if not os.path.exists(train_csv):
        print(f"❌ 训练数据文件不存在: {train_csv}")
        return
    
    try:
        # 创建数据加载器
        data_loaders = create_data_loaders(
            train_csv=train_csv,
            val_csv=val_csv,
            batch_size=4,
            num_workers=0,  # 调试时使用0
            scaffold_modality='smiles'
        )
        
        # 测试训练数据加载器
        train_loader = data_loaders['train']
        print(f"✅ 训练数据加载器创建成功，批数量: {len(train_loader)}")
        
        # 获取一个批次
        batch = next(iter(train_loader))
        print(f"✅ 批数据加载成功:")
        print(f"   批大小: {batch['batch_size']}")
        print(f"   Scaffold模态: {batch['scaffold_modality']}")
        print(f"   示例Scaffold: {batch['scaffold_data'][0][:50]}...")
        print(f"   示例Text: {batch['text_data'][0][:50]}...")
        print(f"   示例Target: {batch['target_smiles'][0][:50]}...")
        
        # 测试验证数据加载器
        val_loader = data_loaders['val']
        val_batch = next(iter(val_loader))
        print(f"✅ 验证数据加载器工作正常，批大小: {val_batch['batch_size']}")
        
    except Exception as e:
        print(f"❌ 数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_dataset()