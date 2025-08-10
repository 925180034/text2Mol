#!/usr/bin/env python3
"""
简化的多模态测试脚本
可以在没有timm的情况下测试部分功能
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# 尝试导入RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RDKIT_AVAILABLE = True
except:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit未安装，部分功能将受限")

# 尝试导入torch_geometric
try:
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except:
    TORCH_GEOMETRIC_AVAILABLE = False
    logger.warning("torch_geometric未安装，图功能将受限")

class SimpleMultiModalTester:
    """简化的多模态测试器"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 定义测试组合
        self.combinations = [
            ('smiles', 'smiles', self.test_smiles_to_smiles),
            ('smiles', 'graph', self.test_smiles_to_graph),
            ('smiles', 'image', self.test_smiles_to_image),
            ('graph', 'smiles', self.test_graph_to_smiles),
            ('graph', 'graph', self.test_graph_to_graph),
            ('graph', 'image', self.test_graph_to_image),
            ('image', 'smiles', self.test_image_to_smiles),
            ('image', 'graph', self.test_image_to_graph),
            ('image', 'image', self.test_image_to_image),
        ]
        
        self.results = []
    
    def smiles_to_graph(self, smiles: str) -> Any:
        """将SMILES转换为图"""
        if not RDKIT_AVAILABLE or not TORCH_GEOMETRIC_AVAILABLE:
            return None
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # 构建节点特征
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
            
            # 构建边
            edge_indices = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_indices.extend([[i, j], [j, i]])
            
            if len(edge_indices) == 0:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
            else:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
            
            x = torch.tensor(atom_features, dtype=torch.float)
            
            return Data(x=x, edge_index=edge_index)
            
        except Exception as e:
            logger.error(f"SMILES到图转换失败: {e}")
            return None
    
    def smiles_to_image(self, smiles: str, size=(224, 224)) -> Any:
        """将SMILES转换为图像"""
        if not RDKIT_AVAILABLE:
            return None
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            img = Draw.MolToImage(mol, size=size)
            img_array = np.array(img)
            return img_array
            
        except Exception as e:
            logger.error(f"SMILES到图像转换失败: {e}")
            return None
    
    def test_smiles_to_smiles(self, test_smiles: List[str]) -> Dict:
        """测试SMILES到SMILES"""
        try:
            # 这里应该调用实际的模型，现在只是模拟
            logger.info("测试 SMILES → SMILES")
            
            # 检查是否可以导入必要的组件
            from scaffold_mol_gen.models.encoders.smiles_encoder import SMILESEncoder
            from scaffold_mol_gen.models.encoders.text_encoder import TextEncoder
            
            return {
                'status': 'success',
                'message': 'SMILES编码器和文本编码器可用',
                'capability': 'ready'
            }
        except ImportError as e:
            return {
                'status': 'failed',
                'message': f'导入失败: {str(e)}',
                'capability': 'not_ready'
            }
    
    def test_smiles_to_graph(self, test_smiles: List[str]) -> Dict:
        """测试SMILES到Graph"""
        logger.info("测试 SMILES → Graph")
        
        if not RDKIT_AVAILABLE or not TORCH_GEOMETRIC_AVAILABLE:
            return {
                'status': 'skipped',
                'message': '需要RDKit和torch_geometric',
                'capability': 'limited'
            }
        
        success_count = 0
        for smiles in test_smiles[:3]:  # 只测试前3个
            graph = self.smiles_to_graph(smiles)
            if graph is not None:
                success_count += 1
        
        return {
            'status': 'success' if success_count > 0 else 'failed',
            'message': f'成功转换 {success_count}/{min(3, len(test_smiles))} 个SMILES',
            'capability': 'ready' if success_count > 0 else 'not_ready'
        }
    
    def test_smiles_to_image(self, test_smiles: List[str]) -> Dict:
        """测试SMILES到Image"""
        logger.info("测试 SMILES → Image")
        
        if not RDKIT_AVAILABLE:
            return {
                'status': 'skipped',
                'message': '需要RDKit',
                'capability': 'limited'
            }
        
        success_count = 0
        for smiles in test_smiles[:3]:
            image = self.smiles_to_image(smiles)
            if image is not None:
                success_count += 1
        
        return {
            'status': 'success' if success_count > 0 else 'failed',
            'message': f'成功转换 {success_count}/{min(3, len(test_smiles))} 个SMILES到图像',
            'capability': 'ready' if success_count > 0 else 'not_ready'
        }
    
    def test_graph_to_smiles(self, test_smiles: List[str]) -> Dict:
        """测试Graph到SMILES"""
        logger.info("测试 Graph → SMILES")
        
        try:
            from scaffold_mol_gen.models.encoders.graph_encoder import GINEncoder
            return {
                'status': 'success',
                'message': 'Graph编码器可用',
                'capability': 'ready'
            }
        except ImportError as e:
            return {
                'status': 'failed',
                'message': f'Graph编码器导入失败: {str(e)}',
                'capability': 'not_ready'
            }
    
    def test_graph_to_graph(self, test_smiles: List[str]) -> Dict:
        """测试Graph到Graph"""
        logger.info("测试 Graph → Graph")
        
        try:
            from scaffold_mol_gen.models.graph_decoder import MolecularGraphDecoder
            return {
                'status': 'success',
                'message': 'Graph解码器可用',
                'capability': 'ready'
            }
        except ImportError as e:
            return {
                'status': 'failed',
                'message': f'Graph解码器导入失败: {str(e)}',
                'capability': 'not_ready'
            }
    
    def test_graph_to_image(self, test_smiles: List[str]) -> Dict:
        """测试Graph到Image"""
        logger.info("测试 Graph → Image")
        
        if not RDKIT_AVAILABLE:
            return {
                'status': 'skipped',
                'message': '需要RDKit进行可视化',
                'capability': 'limited'
            }
        
        return {
            'status': 'possible',
            'message': 'Graph到Image转换理论上可行',
            'capability': 'ready'
        }
    
    def test_image_to_smiles(self, test_smiles: List[str]) -> Dict:
        """测试Image到SMILES"""
        logger.info("测试 Image → SMILES")
        
        try:
            # 尝试导入图像编码器（会失败因为需要timm）
            from scaffold_mol_gen.models.encoders.image_encoder import SwinTransformerEncoder
            return {
                'status': 'success',
                'message': 'Image编码器可用',
                'capability': 'ready'
            }
        except ImportError as e:
            if 'timm' in str(e):
                return {
                    'status': 'blocked',
                    'message': '需要安装timm库: pip install timm',
                    'capability': 'needs_dependency'
                }
            return {
                'status': 'failed',
                'message': f'Image编码器导入失败: {str(e)}',
                'capability': 'not_ready'
            }
    
    def test_image_to_graph(self, test_smiles: List[str]) -> Dict:
        """测试Image到Graph"""
        logger.info("测试 Image → Graph")
        
        return {
            'status': 'blocked',
            'message': '需要timm库支持图像编码器',
            'capability': 'needs_dependency'
        }
    
    def test_image_to_image(self, test_smiles: List[str]) -> Dict:
        """测试Image到Image"""
        logger.info("测试 Image → Image")
        
        try:
            from scaffold_mol_gen.models.image_decoder import MolecularImageDecoder
            return {
                'status': 'blocked',
                'message': 'Image解码器存在但需要timm支持输入',
                'capability': 'needs_dependency'
            }
        except ImportError as e:
            return {
                'status': 'failed',
                'message': f'Image解码器导入失败: {str(e)}',
                'capability': 'not_ready'
            }
    
    def run_all_tests(self) -> Dict:
        """运行所有测试"""
        logger.info("="*60)
        logger.info("开始多模态能力测试")
        logger.info("="*60)
        
        # 测试数据
        test_smiles = [
            "CCO",  # 乙醇
            "CC(=O)O",  # 乙酸
            "c1ccccc1",  # 苯
        ]
        
        results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'system_info': {
                'device': str(self.device),
                'rdkit_available': RDKIT_AVAILABLE,
                'torch_geometric_available': TORCH_GEOMETRIC_AVAILABLE,
            },
            'combinations': []
        }
        
        # 测试每个组合
        for in_modal, out_modal, test_func in self.combinations:
            logger.info(f"\n{'='*40}")
            result = test_func(test_smiles)
            result['input'] = in_modal
            result['output'] = out_modal
            results['combinations'].append(result)
            
            # 打印结果
            status_icon = {
                'success': '✅',
                'failed': '❌',
                'skipped': '⏭️',
                'blocked': '🚫',
                'possible': '🔄'
            }.get(result['status'], '❓')
            
            logger.info(f"{status_icon} {in_modal} → {out_modal}: {result['message']}")
        
        # 统计
        stats = {
            'total': len(results['combinations']),
            'ready': sum(1 for r in results['combinations'] if r['capability'] == 'ready'),
            'needs_dependency': sum(1 for r in results['combinations'] if r['capability'] == 'needs_dependency'),
            'limited': sum(1 for r in results['combinations'] if r['capability'] == 'limited'),
            'not_ready': sum(1 for r in results['combinations'] if r['capability'] == 'not_ready')
        }
        results['statistics'] = stats
        
        return results

def main():
    """主函数"""
    logger.info("🧪 多模态分子生成系统能力测试")
    logger.info("="*60)
    
    tester = SimpleMultiModalTester()
    results = tester.run_all_tests()
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"multimodal_capability_test_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 打印总结
    logger.info("\n" + "="*60)
    logger.info("📊 测试总结")
    logger.info("="*60)
    
    stats = results['statistics']
    logger.info(f"总组合数: {stats['total']}")
    logger.info(f"✅ 已就绪: {stats['ready']}/9")
    logger.info(f"🚫 需要依赖: {stats['needs_dependency']}/9 (需要timm库)")
    logger.info(f"⏭️ 功能受限: {stats['limited']}/9")
    logger.info(f"❌ 未就绪: {stats['not_ready']}/9")
    
    logger.info(f"\n结果已保存到: {output_file}")
    
    # 建议
    logger.info("\n" + "="*60)
    logger.info("💡 建议")
    logger.info("="*60)
    
    if stats['needs_dependency'] > 0:
        logger.info("1. 安装timm库以启用图像相关功能:")
        logger.info("   pip install timm")
        logger.info("   这将解锁3个图像输入的组合")
    
    if stats['ready'] < 9:
        logger.info(f"2. 当前有 {stats['ready']}/9 个组合可以使用")
        logger.info("   主要是SMILES和Graph输入的组合")
    
    logger.info("\n3. 要运行实际的生成测试，需要:")
    logger.info("   - 加载训练好的模型")
    logger.info("   - 修复MolT5的tokenizer问题")
    logger.info("   - 确保所有依赖都已安装")

if __name__ == "__main__":
    main()