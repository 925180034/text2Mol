#!/usr/bin/env python3
"""
修复Graph输入处理问题
解决批处理和数据格式问题
"""

import torch
from torch_geometric.data import Data, Batch
from rdkit import Chem
from typing import List, Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedGraphProcessor:
    """修复的图处理器"""
    
    @staticmethod
    def smiles_to_graph(smiles: str) -> Optional[Data]:
        """
        将SMILES转换为图数据
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # 构建原子特征
            atom_features = []
            for atom in mol.GetAtoms():
                features = [
                    float(atom.GetAtomicNum()),
                    float(atom.GetDegree()),
                    float(atom.GetFormalCharge()),
                    float(atom.GetHybridization()),
                    float(atom.GetIsAromatic()),
                    float(atom.GetTotalNumHs()),
                    float(atom.GetNumRadicalElectrons()),
                    float(atom.IsInRing()),
                    float(atom.GetChiralTag()),
                    float(atom.GetMass())
                ]
                atom_features.append(features)
            
            # 构建边索引和边特征
            edge_indices = []
            edge_attrs = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_indices.extend([[i, j], [j, i]])
                
                bond_features = [
                    float(bond.GetBondTypeAsDouble()),
                    float(bond.GetIsAromatic()),
                    float(bond.GetIsConjugated()),
                    float(bond.IsInRing()),
                    float(bond.GetStereo())
                ]
                edge_attrs.extend([bond_features, bond_features])
            
            # 处理没有边的情况（单原子分子）
            if len(edge_indices) == 0:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, 5), dtype=torch.float)
            else:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
            
            x = torch.tensor(atom_features, dtype=torch.float)
            
            # 创建图数据对象
            graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            
            return graph
            
        except Exception as e:
            logger.error(f"SMILES到图转换失败 ({smiles}): {e}")
            return None
    
    @staticmethod
    def create_default_graph() -> Data:
        """创建默认的图（单个碳原子）"""
        # 单个碳原子的特征
        x = torch.tensor([[6.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 12.01]], dtype=torch.float)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 5), dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    @staticmethod
    def prepare_graph_batch(smiles_list: List[str], device: str = 'cuda') -> Optional[Batch]:
        """
        准备图批处理数据
        
        Args:
            smiles_list: SMILES字符串列表
            device: 设备
            
        Returns:
            批处理的图数据
        """
        graphs = []
        
        for smiles in smiles_list:
            graph = FixedGraphProcessor.smiles_to_graph(smiles)
            if graph is None:
                # 使用默认图作为后备
                logger.warning(f"使用默认图替代无效的SMILES: {smiles}")
                graph = FixedGraphProcessor.create_default_graph()
            graphs.append(graph)
        
        if not graphs:
            logger.error("没有有效的图数据")
            return None
        
        # 批处理并移动到设备
        try:
            batch = Batch.from_data_list(graphs)
            batch = batch.to(device)
            
            # 验证批处理结果
            if not hasattr(batch, 'x'):
                logger.error("批处理后缺少x属性")
                return None
            
            logger.info(f"成功创建图批处理: {len(graphs)}个图, 总节点数={batch.x.shape[0]}")
            return batch
            
        except Exception as e:
            logger.error(f"图批处理失败: {e}")
            return None
    
    @staticmethod
    def process_graph_input(input_data: Union[List[str], List[Data], Batch], 
                          device: str = 'cuda') -> Optional[Batch]:
        """
        处理各种格式的图输入
        
        Args:
            input_data: SMILES列表、图列表或已批处理的图
            device: 设备
            
        Returns:
            批处理的图数据
        """
        if isinstance(input_data, Batch):
            # 已经是批处理数据
            return input_data.to(device)
        
        elif isinstance(input_data, list):
            if len(input_data) == 0:
                logger.error("输入列表为空")
                return None
            
            if isinstance(input_data[0], str):
                # SMILES列表
                return FixedGraphProcessor.prepare_graph_batch(input_data, device)
            
            elif isinstance(input_data[0], Data):
                # 图数据列表
                try:
                    batch = Batch.from_data_list(input_data)
                    return batch.to(device)
                except Exception as e:
                    logger.error(f"图列表批处理失败: {e}")
                    return None
            
            else:
                logger.error(f"不支持的列表元素类型: {type(input_data[0])}")
                return None
        
        else:
            logger.error(f"不支持的输入类型: {type(input_data)}")
            return None


def test_graph_processor():
    """测试图处理器"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    
    # 测试SMILES
    test_smiles = [
        "CCO",  # 乙醇
        "CC(=O)O",  # 乙酸
        "c1ccccc1",  # 苯
        "INVALID",  # 无效的SMILES
        "",  # 空字符串
    ]
    
    logger.info(f"\n测试{len(test_smiles)}个SMILES...")
    
    # 测试单个转换
    logger.info("\n1. 测试单个SMILES转换:")
    for smiles in test_smiles[:3]:
        graph = FixedGraphProcessor.smiles_to_graph(smiles)
        if graph:
            logger.info(f"  {smiles}: 节点数={graph.x.shape[0]}, 边数={graph.edge_index.shape[1]//2}")
        else:
            logger.info(f"  {smiles}: 转换失败")
    
    # 测试批处理
    logger.info("\n2. 测试批处理:")
    batch = FixedGraphProcessor.prepare_graph_batch(test_smiles, device)
    if batch:
        logger.info(f"  批处理成功: 总节点数={batch.x.shape[0]}, 批次大小={batch.num_graphs}")
        logger.info(f"  批处理属性: {list(batch.keys())}")
    else:
        logger.info("  批处理失败")
    
    # 测试不同输入格式
    logger.info("\n3. 测试不同输入格式:")
    
    # 测试SMILES列表
    result1 = FixedGraphProcessor.process_graph_input(test_smiles[:3], device)
    logger.info(f"  SMILES列表: {'成功' if result1 else '失败'}")
    
    # 测试图列表
    graphs = [FixedGraphProcessor.smiles_to_graph(s) for s in test_smiles[:3]]
    graphs = [g for g in graphs if g is not None]
    result2 = FixedGraphProcessor.process_graph_input(graphs, device)
    logger.info(f"  图列表: {'成功' if result2 else '失败'}")
    
    # 测试已批处理的数据
    if batch:
        result3 = FixedGraphProcessor.process_graph_input(batch, device)
        logger.info(f"  已批处理数据: {'成功' if result3 else '失败'}")
    
    logger.info("\n测试完成!")
    
    return batch


if __name__ == "__main__":
    test_graph_processor()