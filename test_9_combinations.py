#!/usr/bin/env python3
"""
测试9种输入输出组合
验证整个多模态系统的功能
"""

import os
import sys
import torch
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from scaffold_mol_gen.models.end2end_model import End2EndMolecularGenerator
from scaffold_mol_gen.models.output_decoders import OutputDecoder
from scaffold_mol_gen.models.encoders.graph_encoder import GraphFeatureExtractor

logger = logging.getLogger(__name__)

class NineCombinationTester:
    """9种I/O组合测试器"""
    
    def __init__(self, 
                 model_path: str = None,
                 device: str = 'cuda'):
        """
        Args:
            model_path: 预训练模型路径（可选）
            device: 计算设备
        """
        self.device = device
        self.output_decoder = OutputDecoder()
        
        # 定义9种组合
        self.combinations = [
            ('smiles', 'smiles'),  # 1. SMILES + Text → SMILES
            ('smiles', 'graph'),   # 2. SMILES + Text → Graph  
            ('smiles', 'image'),   # 3. SMILES + Text → Image
            ('graph', 'smiles'),   # 4. Graph + Text → SMILES
            ('graph', 'graph'),    # 5. Graph + Text → Graph
            ('graph', 'image'),    # 6. Graph + Text → Image
            ('image', 'smiles'),   # 7. Image + Text → SMILES
            ('image', 'graph'),    # 8. Image + Text → Graph
            ('image', 'image'),    # 9. Image + Text → Image
        ]
        
        # 创建模型
        self.model = self._create_model()
        
        # 加载预训练权重（如果提供）
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
            logger.info(f"加载预训练模型: {model_path}")
        else:
            logger.info("使用随机初始化的模型进行测试")
    
    def _create_model(self) -> End2EndMolecularGenerator:
        """创建端到端模型"""
        
        # MolT5路径
        molt5_path = "/root/autodl-tmp/text2Mol-models/molt5-base"
        
        if not Path(molt5_path).exists():
            logger.warning(f"MolT5模型未找到: {molt5_path}")
            logger.info("将尝试从HuggingFace下载...")
            molt5_path = "laituan245/molt5-base"
        
        # 创建模型
        model = End2EndMolecularGenerator(
            hidden_size=768,
            molt5_path=molt5_path,
            use_scibert=False,
            freeze_encoders=True,
            freeze_molt5=True,
            fusion_type='both',
            device=self.device
        )
        
        return model.to(self.device)
    
    def _load_model(self, model_path: str):
        """加载预训练模型"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 加载不同组件的权重
            if 'e2e_model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['e2e_model_state_dict'], strict=False)
            elif 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
            
            logger.info("模型权重加载成功")
            
        except Exception as e:
            logger.warning(f"模型权重加载失败: {e}")
            logger.info("继续使用随机初始化模型")
    
    def _prepare_test_data(self) -> Dict[str, Any]:
        """准备测试数据"""
        
        # 测试SMILES
        test_smiles = [
            "c1ccccc1",  # 苯环
            "CCO",       # 乙醇
            "CC(=O)O",   # 乙酸
        ]
        
        # 测试文本描述
        test_texts = [
            "Aromatic ring compound",
            "Simple alcohol",
            "Carboxylic acid",
        ]
        
        # 转换为其他模态
        test_graphs = []
        test_images = []
        
        for smiles in test_smiles:
            # 转换为图
            graph = self.output_decoder.decode(smiles, 'graph')
            test_graphs.append(graph)
            
            # 转换为图像（转换为tensor格式）
            image_array = self.output_decoder.image_decoder.decode(smiles, size=(224, 224))  # 匹配模型尺寸
            if image_array is not None:
                test_images.append(torch.from_numpy(image_array))
            else:
                # 创建空白图像作为占位符
                test_images.append(torch.zeros(3, 224, 224))  # 使用224x224
        
        return {
            'smiles': test_smiles,
            'texts': test_texts,
            'graphs': test_graphs,
            'images': test_images
        }
    
    def test_single_combination(self, 
                               scaffold_modality: str,
                               output_modality: str,
                               test_data: Dict[str, Any],
                               test_idx: int = 0) -> Dict[str, Any]:
        """测试单个I/O组合"""
        
        logger.info(f"测试: {scaffold_modality.upper()} + Text → {output_modality.upper()}")
        
        # 准备输入数据
        if scaffold_modality == 'smiles':
            scaffold_data = test_data['smiles'][test_idx]
        elif scaffold_modality == 'graph':
            scaffold_data = test_data['graphs'][test_idx]
        elif scaffold_modality == 'image':
            scaffold_data = test_data['images'][test_idx]
        else:
            raise ValueError(f"不支持的scaffold模态: {scaffold_modality}")
        
        text_data = test_data['texts'][test_idx]
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 生成输出
            with torch.no_grad():
                output = self.model.generate(
                    scaffold_data=scaffold_data,
                    text_data=text_data,
                    scaffold_modality=scaffold_modality,
                    output_modality=output_modality,
                    num_beams=3,
                    temperature=0.8,
                    max_length=128,
                    num_return_sequences=1
                )
            
            # 记录结束时间
            end_time = time.time()
            generation_time = end_time - start_time
            
            # 验证输出
            success, validation_msg = self._validate_output(output, output_modality)
            
            result = {
                'success': success,
                'output': output,
                'generation_time': generation_time,
                'validation_message': validation_msg,
                'input_scaffold': scaffold_data,
                'input_text': text_data
            }
            
            logger.info(f"  成功: {success}")
            logger.info(f"  时间: {generation_time:.3f}s")
            logger.info(f"  验证: {validation_msg}")
            
            return result
            
        except Exception as e:
            logger.error(f"  失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'generation_time': time.time() - start_time,
                'input_scaffold': scaffold_data,
                'input_text': text_data
            }
    
    def _validate_output(self, output: Any, modality: str) -> Tuple[bool, str]:
        """验证输出结果"""
        
        if output is None:
            return False, "输出为None"
        
        if modality == 'smiles':
            if isinstance(output, list) and len(output) > 0:
                smiles = output[0]
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    return True, f"有效SMILES: {smiles}"
                else:
                    return False, f"无效SMILES: {smiles}"
            else:
                return False, "SMILES输出格式错误"
        
        elif modality == 'graph':
            if isinstance(output, list) and len(output) > 0:
                from torch_geometric.data import Data
                graph = output[0]
                if isinstance(graph, Data):
                    num_nodes = graph.x.shape[0] if graph.x is not None else 0
                    num_edges = graph.edge_index.shape[1] if graph.edge_index is not None else 0
                    return True, f"图结构: {num_nodes}个节点, {num_edges}条边"
                else:
                    return False, "图输出格式错误"
            else:
                return False, "图输出为空"
        
        elif modality == 'image':
            if isinstance(output, list) and len(output) > 0:
                from PIL import Image
                image = output[0]
                if isinstance(image, Image.Image):
                    return True, f"图像: {image.size[0]}x{image.size[1]}"
                elif torch.is_tensor(image):
                    return True, f"图像张量: {image.shape}"
                else:
                    return False, "图像格式错误"
            else:
                return False, "图像输出为空"
        
        return False, "未知输出模态"
    
    def test_all_combinations(self, save_results: bool = True) -> Dict[str, Any]:
        """测试所有9种组合"""
        
        logger.info("开始测试所有9种输入输出组合")
        logger.info("=" * 60)
        
        # 准备测试数据
        test_data = self._prepare_test_data()
        
        # 结果统计
        results = {}
        success_count = 0
        total_time = 0
        
        # 逐个测试每种组合
        for i, (scaffold_mod, output_mod) in enumerate(self.combinations, 1):
            logger.info(f"\n[{i}/9] 测试组合: {scaffold_mod} + text → {output_mod}")
            logger.info("-" * 40)
            
            # 测试组合
            result = self.test_single_combination(
                scaffold_modality=scaffold_mod,
                output_modality=output_mod,
                test_data=test_data,
                test_idx=0  # 使用第一个测试样本
            )
            
            # 记录结果
            combination_key = f"{scaffold_mod}2{output_mod}"
            results[combination_key] = result
            
            if result['success']:
                success_count += 1
            
            total_time += result.get('generation_time', 0)
        
        # 汇总结果
        summary = {
            'total_combinations': len(self.combinations),
            'successful_combinations': success_count,
            'success_rate': success_count / len(self.combinations),
            'total_time': total_time,
            'average_time': total_time / len(self.combinations),
            'detailed_results': results
        }
        
        # 显示汇总
        logger.info(f"\n{'='*60}")
        logger.info("测试结果汇总")
        logger.info(f"{'='*60}")
        logger.info(f"总组合数: {summary['total_combinations']}")
        logger.info(f"成功组合数: {summary['successful_combinations']}")
        logger.info(f"成功率: {summary['success_rate']:.1%}")
        logger.info(f"总耗时: {summary['total_time']:.2f}s")
        logger.info(f"平均耗时: {summary['average_time']:.3f}s/组合")
        
        # 详细结果
        logger.info(f"\n详细结果:")
        for i, (scaffold_mod, output_mod) in enumerate(self.combinations, 1):
            combination_key = f"{scaffold_mod}2{output_mod}"
            result = results[combination_key]
            status = "✅" if result['success'] else "❌"
            time_info = f"{result.get('generation_time', 0):.3f}s"
            logger.info(f"{i:2d}. {scaffold_mod:6s} + text → {output_mod:6s} {status} ({time_info})")
        
        # 保存结果
        if save_results:
            self._save_results(summary)
        
        return summary
    
    def _save_results(self, summary: Dict[str, Any]):
        """保存测试结果"""
        import json
        from datetime import datetime
        
        # 创建结果目录
        results_dir = Path("test_results")
        results_dir.mkdir(exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"9combinations_test_{timestamp}.json"
        filepath = results_dir / filename
        
        # 处理不可序列化的对象
        serializable_summary = {}
        for key, value in summary.items():
            if key == 'detailed_results':
                serializable_detailed = {}
                for combo_key, combo_result in value.items():
                    serializable_combo = {}
                    for k, v in combo_result.items():
                        if k in ['input_scaffold', 'output']:
                            # 跳过复杂对象，只保留基本信息
                            serializable_combo[k] = str(v)[:100] + "..." if len(str(v)) > 100 else str(v)
                        else:
                            serializable_combo[k] = v
                    serializable_detailed[combo_key] = serializable_combo
                serializable_summary[key] = serializable_detailed
            else:
                serializable_summary[key] = value
        
        # 保存JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"测试结果已保存: {filepath}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="测试9种输入输出组合")
    parser.add_argument('--model-path', type=str, help='预训练模型路径')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')
    parser.add_argument('--log-level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA不可用，切换到CPU")
        args.device = 'cpu'
    
    logger.info(f"使用设备: {args.device}")
    
    try:
        # 创建测试器
        tester = NineCombinationTester(
            model_path=args.model_path,
            device=args.device
        )
        
        # 运行测试
        results = tester.test_all_combinations(save_results=True)
        
        # 显示最终结果
        success_rate = results['success_rate']
        if success_rate >= 0.8:
            logger.info(f"🎉 测试完成！成功率: {success_rate:.1%} (优秀)")
        elif success_rate >= 0.6:
            logger.info(f"✅ 测试完成！成功率: {success_rate:.1%} (良好)")
        elif success_rate >= 0.4:
            logger.info(f"⚠️ 测试完成！成功率: {success_rate:.1%} (一般)")
        else:
            logger.info(f"❌ 测试完成！成功率: {success_rate:.1%} (需要改进)")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        raise


if __name__ == "__main__":
    main()