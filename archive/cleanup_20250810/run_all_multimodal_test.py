#!/usr/bin/env python3
"""
完整的多模态分子生成系统测试
测试所有9种输入输出组合
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入必要的组件
from scaffold_mol_gen.models.end2end_model import End2EndMolecularGenerator
from scaffold_mol_gen.data.multimodal_preprocessor import MultiModalPreprocessor
# 简化的评估指标
def calculate_validity(smiles_list):
    from rdkit import Chem
    valid = sum(1 for s in smiles_list if Chem.MolFromSmiles(s) is not None)
    return valid / len(smiles_list) if smiles_list else 0

def calculate_uniqueness(smiles_list):
    unique = len(set(smiles_list))
    return unique / len(smiles_list) if smiles_list else 0

def calculate_novelty(generated, reference):
    novel = sum(1 for s in generated if s not in reference)
    return novel / len(generated) if generated else 0
import torchvision.transforms as transforms
from torch_geometric.data import Batch

class MultiModalEvaluator:
    """多模态评估器"""
    
    def __init__(self, model_path: str = None, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 加载模型
        if model_path and Path(model_path).exists():
            logger.info(f"加载模型: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model = checkpoint.get('model', End2EndMolecularGenerator())
            self.model.to(self.device)
            self.model.eval()
        else:
            logger.info("创建新模型实例")
            self.model = End2EndMolecularGenerator(device=str(self.device))
            self.model.to(self.device)
            self.model.eval()
        
        # 初始化预处理器
        self.preprocessor = MultiModalPreprocessor()
        
        # 定义所有9种组合
        self.combinations = [
            ('smiles', 'smiles', '✅'),
            ('smiles', 'graph', '✅'),
            ('smiles', 'image', '✅'),
            ('graph', 'smiles', '✅'),
            ('graph', 'graph', '✅'),
            ('graph', 'image', '✅'),
            ('image', 'smiles', '✅'),
            ('image', 'graph', '✅'),
            ('image', 'image', '✅'),
        ]
        
        # 图像转换
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    
    def prepare_scaffold_data(self, smiles_list: List[str], modality: str) -> Any:
        """准备不同模态的scaffold数据"""
        
        if modality == 'smiles':
            return smiles_list
        
        elif modality == 'graph':
            graphs = []
            for smiles in smiles_list:
                graph = self.preprocessor.smiles_to_graph(smiles)
                if graph is not None:
                    graphs.append(graph)
                else:
                    # 创建一个简单的默认图
                    logger.warning(f"无法转换SMILES到图: {smiles}")
                    # 使用一个简单的碳原子作为默认
                    import torch
                    from torch_geometric.data import Data
                    x = torch.tensor([[6, 0, 0, 0, 0, 0, 0, 0, 0, 12.01]], dtype=torch.float)
                    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                    edge_attr = torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.float)
                    graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
            
            # 批处理图数据
            if graphs:
                return Batch.from_data_list(graphs).to(self.device)
            return None
        
        elif modality == 'image':
            images = []
            for smiles in smiles_list:
                image = self.preprocessor.smiles_to_image(smiles)
                if image is not None:
                    if isinstance(image, np.ndarray):
                        # 转换为tensor
                        image_tensor = self.image_transform(image)
                        images.append(image_tensor)
                else:
                    # 创建默认的白色图像
                    logger.warning(f"无法转换SMILES到图像: {smiles}")
                    default_img = np.ones((224, 224, 3), dtype=np.uint8) * 255
                    image_tensor = self.image_transform(default_img)
                    images.append(image_tensor)
            
            # 批处理图像
            if images:
                return torch.stack(images).to(self.device)
            return None
        
        else:
            raise ValueError(f"不支持的模态: {modality}")
    
    def test_single_combination(self, 
                               scaffold_modality: str,
                               output_modality: str,
                               test_smiles: List[str],
                               test_texts: List[str]) -> Dict:
        """测试单个输入输出组合"""
        
        logger.info(f"\n测试组合: {scaffold_modality} → {output_modality}")
        result = {
            'scaffold_modality': scaffold_modality,
            'output_modality': output_modality,
            'status': 'testing',
            'metrics': {},
            'examples': [],
            'error': None
        }
        
        try:
            # 准备输入数据
            scaffold_data = self.prepare_scaffold_data(test_smiles, scaffold_modality)
            if scaffold_data is None:
                raise ValueError(f"无法准备{scaffold_modality}模态数据")
            
            # 生成输出
            with torch.no_grad():
                if output_modality == 'smiles':
                    # 直接生成SMILES
                    generated = self.model.generate(
                        scaffold_data=scaffold_data,
                        text_data=test_texts,
                        scaffold_modality=scaffold_modality,
                        output_modality='smiles',
                        num_beams=3,
                        temperature=0.8,
                        max_length=128
                    )
                elif output_modality in ['graph', 'image']:
                    # 先生成SMILES，再转换
                    smiles_output = self.model.generate(
                        scaffold_data=scaffold_data,
                        text_data=test_texts,
                        scaffold_modality=scaffold_modality,
                        output_modality='smiles',
                        num_beams=3,
                        temperature=0.8,
                        max_length=128
                    )
                    
                    # 转换到目标模态
                    if output_modality == 'graph':
                        generated = []
                        for smi in smiles_output:
                            graph = self.preprocessor.smiles_to_graph(smi)
                            generated.append(graph)
                    else:  # image
                        generated = []
                        for smi in smiles_output:
                            image = self.preprocessor.smiles_to_image(smi)
                            generated.append(image)
                else:
                    raise ValueError(f"不支持的输出模态: {output_modality}")
            
            # 计算指标（仅对SMILES输出）
            if output_modality == 'smiles' and isinstance(generated, list):
                result['metrics'] = {
                    'validity': calculate_validity(generated),
                    'uniqueness': calculate_uniqueness(generated),
                    'novelty': calculate_novelty(generated, test_smiles),
                    'samples_generated': len(generated)
                }
            elif output_modality == 'graph':
                valid_graphs = sum(1 for g in generated if g is not None)
                result['metrics'] = {
                    'valid_graphs': valid_graphs,
                    'total_graphs': len(generated),
                    'success_rate': valid_graphs / len(generated) if generated else 0
                }
            elif output_modality == 'image':
                valid_images = sum(1 for img in generated if img is not None)
                result['metrics'] = {
                    'valid_images': valid_images,
                    'total_images': len(generated),
                    'success_rate': valid_images / len(generated) if generated else 0
                }
            
            # 保存示例
            for i in range(min(3, len(test_smiles))):
                example = {
                    'input_smiles': test_smiles[i],
                    'input_text': test_texts[i][:100] + '...' if len(test_texts[i]) > 100 else test_texts[i],
                    'input_modality': scaffold_modality,
                    'output_modality': output_modality
                }
                
                if output_modality == 'smiles' and i < len(generated):
                    example['generated'] = generated[i] if isinstance(generated[i], str) else str(generated[i])
                elif output_modality == 'graph' and i < len(generated) and generated[i]:
                    example['generated'] = f"Graph(nodes={generated[i].x.shape[0]}, edges={generated[i].edge_index.shape[1]//2})"
                elif output_modality == 'image' and i < len(generated) and generated[i] is not None:
                    example['generated'] = f"Image(shape={generated[i].shape if hasattr(generated[i], 'shape') else 'unknown'})"
                
                result['examples'].append(example)
            
            result['status'] = 'success'
            logger.info(f"  ✅ 成功 - {result['metrics']}")
            
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
            logger.error(f"  ❌ 失败: {e}")
        
        return result
    
    def run_all_tests(self, test_data_path: str = None, sample_size: int = 10):
        """运行所有组合测试"""
        
        # 准备测试数据
        if test_data_path and Path(test_data_path).exists():
            logger.info(f"加载测试数据: {test_data_path}")
            test_df = pd.read_csv(test_data_path).head(sample_size)
            test_smiles = test_df['SMILES'].tolist() if 'SMILES' in test_df else test_df.iloc[:, 0].tolist()
            test_texts = test_df['description'].tolist() if 'description' in test_df else test_df.iloc[:, 1].tolist()
        else:
            logger.info("使用示例数据进行测试")
            test_smiles = [
                "CCO",  # 乙醇
                "CC(=O)O",  # 乙酸
                "c1ccccc1",  # 苯
                "CC(C)CC(C)(C)O",  # 复杂分子
                "O=C(O)COc1ccc(Cl)c2cccnc12",  # 更复杂的分子
            ]
            test_texts = [
                "A simple alcohol molecule",
                "An organic acid",
                "An aromatic ring",
                "A branched alcohol",
                "A complex heterocyclic compound"
            ] * (sample_size // 5 + 1)
            test_smiles = test_smiles * (sample_size // 5 + 1)
            test_smiles = test_smiles[:sample_size]
            test_texts = test_texts[:sample_size]
        
        logger.info(f"测试样本数: {len(test_smiles)}")
        
        # 测试所有组合
        all_results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_info': {
                'device': str(self.device),
                'sample_size': len(test_smiles)
            },
            'combinations': []
        }
        
        for scaffold_mod, output_mod, status_icon in self.combinations:
            logger.info(f"\n{'='*60}")
            logger.info(f"测试组合 {status_icon}: {scaffold_mod} → {output_mod}")
            logger.info(f"{'='*60}")
            
            result = self.test_single_combination(
                scaffold_modality=scaffold_mod,
                output_modality=output_mod,
                test_smiles=test_smiles,
                test_texts=test_texts
            )
            
            all_results['combinations'].append(result)
        
        # 统计结果
        success_count = sum(1 for c in all_results['combinations'] if c['status'] == 'success')
        failed_count = sum(1 for c in all_results['combinations'] if c['status'] == 'failed')
        
        all_results['summary'] = {
            'total_combinations': len(self.combinations),
            'successful': success_count,
            'failed': failed_count,
            'success_rate': success_count / len(self.combinations) if self.combinations else 0
        }
        
        return all_results

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='测试所有多模态组合')
    parser.add_argument('--model-path', type=str, 
                      default='/root/autodl-tmp/text2Mol-outputs/optimized_20250809_105726/best_model.pt',
                      help='模型路径')
    parser.add_argument('--test-file', type=str,
                      default='Datasets/test.csv',
                      help='测试数据文件')
    parser.add_argument('--sample-size', type=int, default=10,
                      help='测试样本数')
    parser.add_argument('--device', type=str, default='cuda',
                      help='设备')
    parser.add_argument('--output-dir', type=str, default='.',
                      help='输出目录')
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("🧪 多模态分子生成系统全面测试")
    logger.info("="*70)
    logger.info(f"测试所有9种输入输出组合")
    logger.info(f"每种组合测试{args.sample_size}个样本")
    logger.info("="*70)
    
    # 创建评估器
    evaluator = MultiModalEvaluator(
        model_path=args.model_path,
        device=args.device
    )
    
    # 运行测试
    results = evaluator.run_all_tests(
        test_data_path=args.test_file,
        sample_size=args.sample_size
    )
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(args.output_dir) / f"multimodal_test_results_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    logger.info(f"\n{'='*70}")
    logger.info("📊 测试完成")
    logger.info(f"{'='*70}")
    logger.info(f"总组合数: {results['summary']['total_combinations']}")
    logger.info(f"成功: {results['summary']['successful']}")
    logger.info(f"失败: {results['summary']['failed']}")
    logger.info(f"成功率: {results['summary']['success_rate']:.1%}")
    logger.info(f"\n结果已保存到: {output_file}")
    
    # 打印详细结果
    logger.info(f"\n{'='*70}")
    logger.info("📋 详细结果")
    logger.info(f"{'='*70}")
    
    for combo in results['combinations']:
        status_icon = "✅" if combo['status'] == 'success' else "❌"
        logger.info(f"{status_icon} {combo['scaffold_modality']}→{combo['output_modality']}: {combo['status']}")
        if combo['metrics']:
            for key, value in combo['metrics'].items():
                if isinstance(value, float):
                    logger.info(f"    {key}: {value:.4f}")
                else:
                    logger.info(f"    {key}: {value}")
        if combo['error']:
            logger.info(f"    错误: {combo['error']}")

if __name__ == "__main__":
    main()