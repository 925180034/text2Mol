"""
实际的Graph和Image输出解码器
将SMILES转换为Graph和Image表示
"""
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.data import Data
from PIL import Image
from typing import List, Optional, Union, Tuple

class SMILESToGraphDecoder:
    """SMILES到Graph的实际解码器"""
    
    @staticmethod
    def decode(smiles: Union[str, List[str]]) -> Union[Data, List[Data]]:
        """
        将SMILES转换为图表示
        
        Args:
            smiles: SMILES字符串或列表
            
        Returns:
            PyTorch Geometric的图数据
        """
        if isinstance(smiles, str):
            return SMILESToGraphDecoder._decode_single(smiles)
        else:
            return [SMILESToGraphDecoder._decode_single(s) for s in smiles]
    
    @staticmethod
    def _decode_single(smiles: str) -> Optional[Data]:
        """解码单个SMILES"""
        try:
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
                    int(atom.GetIsAromatic()),
                    atom.GetTotalNumHs(),
                    atom.GetNumRadicalElectrons(),
                    int(atom.IsInRing()),
                    int(atom.GetChiralTag()),
                    atom.GetMass()
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
                    bond.GetBondTypeAsDouble(),
                    int(bond.GetIsAromatic()),
                    int(bond.GetIsConjugated()),
                    int(bond.IsInRing()),
                    int(bond.GetStereo())
                ]
                edge_attrs.extend([bond_features, bond_features])
            
            # 处理没有边的情况
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


class SMILESToImageDecoder:
    """SMILES到Image的实际解码器"""
    
    @staticmethod
    def decode(smiles: Union[str, List[str]], size: Tuple[int, int] = (299, 299)) -> Union[np.ndarray, List[np.ndarray]]:
        """
        将SMILES转换为分子图像
        
        Args:
            smiles: SMILES字符串或列表
            size: 图像大小
            
        Returns:
            图像数组 (3, H, W) 或列表
        """
        if isinstance(smiles, str):
            return SMILESToImageDecoder._decode_single(smiles, size)
        else:
            return [SMILESToImageDecoder._decode_single(s, size) for s in smiles]
    
    @staticmethod
    def _decode_single(smiles: str, size: Tuple[int, int] = (299, 299)) -> Optional[np.ndarray]:
        """解码单个SMILES"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # 生成分子图像
            img = Draw.MolToImage(mol, size=size)
            
            # 转换为numpy数组
            img_array = np.array(img)
            
            # 确保是RGB格式
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            
            # 转换为CHW格式 (3, H, W)
            img_array = np.transpose(img_array, (2, 0, 1))
            
            # 归一化到[0, 1]
            img_array = img_array.astype(np.float32) / 255.0
            
            return img_array
            
        except Exception as e:
            print(f"Error converting SMILES to image: {e}")
            return None
    
    @staticmethod
    def save_image(smiles: str, filepath: str, size: Tuple[int, int] = (299, 299)) -> bool:
        """保存分子图像到文件"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            
            img = Draw.MolToImage(mol, size=size)
            img.save(filepath)
            return True
            
        except Exception as e:
            print(f"Error saving image: {e}")
            return False


class OutputDecoder:
    """统一的输出解码器"""
    
    def __init__(self):
        self.graph_decoder = SMILESToGraphDecoder()
        self.image_decoder = SMILESToImageDecoder()
    
    def decode(self, smiles: Union[str, List[str]], output_modality: str = 'smiles'):
        """
        将SMILES解码为指定模态
        
        Args:
            smiles: SMILES字符串或列表
            output_modality: 输出模态 ('smiles', 'graph', 'image')
            
        Returns:
            解码后的数据
        """
        if output_modality == 'smiles':
            return smiles
        elif output_modality == 'graph':
            return self.graph_decoder.decode(smiles)
        elif output_modality == 'image':
            return self.image_decoder.decode(smiles)
        else:
            raise ValueError(f"Unsupported output modality: {output_modality}")


# 测试解码器
if __name__ == "__main__":
    import time
    
    # 测试SMILES列表
    test_smiles_list = [
        "CCO",  # 乙醇
        "CC(=O)O",  # 乙酸
        "c1ccccc1",  # 苯
        "CC(C)CC(C)(C)O",  # 较复杂的分子
    ]
    
    print("="*70)
    print("🧪 测试输出解码器")
    print("="*70)
    
    decoder = OutputDecoder()
    
    for smiles in test_smiles_list:
        print(f"\n测试SMILES: {smiles}")
        
        # 测试Graph解码
        start_time = time.time()
        graph = decoder.decode(smiles, 'graph')
        graph_time = time.time() - start_time
        if graph:
            print(f"  Graph: 节点数={graph.x.shape[0]}, 边数={graph.edge_index.shape[1]//2}, 耗时={graph_time:.3f}s")
        
        # 测试Image解码
        start_time = time.time()
        image = decoder.decode(smiles, 'image')
        image_time = time.time() - start_time
        if image is not None:
            print(f"  Image: 形状={image.shape}, 范围=[{image.min():.2f}, {image.max():.2f}], 耗时={image_time:.3f}s")
    
    # 测试批量解码
    print("\n批量解码测试:")
    start_time = time.time()
    graphs = decoder.decode(test_smiles_list, 'graph')
    print(f"  批量Graph解码: {len(graphs)}个分子, 耗时={time.time()-start_time:.3f}s")
    
    start_time = time.time()
    images = decoder.decode(test_smiles_list, 'image')
    print(f"  批量Image解码: {len(images)}个分子, 耗时={time.time()-start_time:.3f}s")
    
    print("\n✅ 解码器测试完成！")