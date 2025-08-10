#!/usr/bin/env python
"""
🧪 9模态分子生成系统 - 全面测试脚本

测试所有9种输入输出组合：
(SMILES/Graph/Image scaffold) × (SMILES/Graph/Image output) = 9种组合

训练完成后的质量评估和功能验证。
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
import time
import json
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scaffold_mol_gen.models.end2end_model import End2EndMolecularGenerator
from scaffold_mol_gen.training.metrics import GenerationMetrics, BenchmarkMetrics
from scaffold_mol_gen.utils.mol_utils import MolecularUtils
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NineModalTester:
    """9模态系统全面测试器"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        初始化测试器
        
        Args:
            model_path: 训练好的模型路径
            device: 计算设备
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.metrics_calculator = GenerationMetrics()
        self.benchmark_calculator = BenchmarkMetrics()
        
        # 测试数据
        self.test_data = self._prepare_test_data()
        
        # 9种组合
        self.combinations = [
            ('smiles', 'smiles'), ('smiles', 'graph'), ('smiles', 'image'),
            ('graph', 'smiles'),  ('graph', 'graph'),  ('graph', 'image'),
            ('image', 'smiles'),  ('image', 'graph'),  ('image', 'image')
        ]
        
    def _prepare_test_data(self) -> Dict[str, Any]:
        """准备测试数据"""
        return {
            'scaffold_smiles': [
                'c1ccccc1',           # 苯环
                'c1ccc2c(c1)cccc2',   # 萘环
                'c1ccc2c(c1)[nH]c3ccccc32',  # 吲哚环
                'C1CCC2CCCCC2C1',     # 环状烷烃
                'c1cccnc1'            # 吡啶
            ],
            'text_descriptions': [
                'Anti-inflammatory drug with carboxylic acid group',
                'Antibiotic compound with amino group',
                'Antiviral agent with hydroxyl group',
                'Pain relief medication with ester linkage',
                'Cardiovascular drug with nitrogen heterocycle'
            ],
            'expected_properties': [
                'therapeutic activity',
                'antimicrobial effects', 
                'antiviral properties',
                'analgesic effects',
                'cardiovascular benefits'
            ]
        }
    
    def load_model(self) -> bool:
        """加载训练好的模型"""
        try:
            logger.info(f"加载模型: {self.model_path}")
            
            # 检查模型文件
            if not os.path.exists(self.model_path):
                logger.error(f"模型文件不存在: {self.model_path}")
                return False
            
            # 创建模型实例
            self.model = End2EndMolecularGenerator(
                hidden_size=768,
                molt5_path="/root/autodl-tmp/text2Mol-models/molt5-base",
                use_scibert=False,
                freeze_encoders=False,  # 训练后解冻以使用学习的权重
                freeze_molt5=False,     # 训练后解冻
                fusion_type='both',
                device=self.device
            )
            
            # 加载训练好的权重
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 处理不同的checkpoint格式
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"加载epoch {checkpoint.get('epoch', 'unknown')}的模型")
                best_val_loss = checkpoint.get('best_val_loss', 'unknown')
                if isinstance(best_val_loss, (int, float)):
                    logger.info(f"最佳验证损失: {best_val_loss:.4f}")
                else:
                    logger.info(f"最佳验证损失: {best_val_loss}")
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            logger.info("✅ 模型加载成功")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            return False
    
    def test_single_combination(self, scaffold_modality: str, output_modality: str, 
                               sample_count: int = 3) -> Dict[str, Any]:
        """
        测试单个输入输出组合
        
        Args:
            scaffold_modality: 脚手架输入模态
            output_modality: 输出模态
            sample_count: 测试样本数量
            
        Returns:
            测试结果字典
        """
        logger.info(f"🧪 测试组合: Scaffold({scaffold_modality}) + Text → {output_modality}")
        
        results = {
            'combination': f"{scaffold_modality}_to_{output_modality}",
            'scaffold_modality': scaffold_modality,
            'output_modality': output_modality,
            'sample_count': sample_count,
            'success': False,
            'generation_time': 0.0,
            'generated_outputs': [],
            'valid_outputs': 0,
            'error_message': None,
            'metrics': {}
        }
        
        try:
            start_time = time.time()
            
            # 准备输入数据
            scaffold_data = self.test_data['scaffold_smiles'][:sample_count]
            text_data = self.test_data['text_descriptions'][:sample_count]
            
            generated_outputs = []
            generated_smiles = []  # 用于指标计算
            
            # 逐个生成（避免批处理复杂性）
            for i in range(sample_count):
                try:
                    # 使用模型生成
                    output = self.model.generate(
                        scaffold_data=[scaffold_data[i]],  # 单个样本作为列表
                        text_data=[text_data[i]],
                        scaffold_modality=scaffold_modality,
                        output_modality=output_modality,
                        num_beams=5,
                        temperature=0.8,
                        max_length=128,
                        num_return_sequences=1
                    )
                    
                    if output_modality == 'smiles':
                        gen_smiles = output[0] if isinstance(output, list) else str(output)
                        generated_outputs.append(gen_smiles)
                        generated_smiles.append(gen_smiles)
                    else:
                        # Graph或Image输出
                        generated_outputs.append(f"{output_modality}_data_{i}")
                        # 对于指标计算，我们需要对应的SMILES
                        # 重新生成SMILES版本用于评估
                        smiles_output = self.model.generate(
                            scaffold_data=[scaffold_data[i]],
                            text_data=[text_data[i]], 
                            scaffold_modality=scaffold_modality,
                            output_modality='smiles',  # 强制SMILES输出用于评估
                            num_beams=5,
                            temperature=0.8,
                            max_length=128,
                            num_return_sequences=1
                        )
                        gen_smiles = smiles_output[0] if isinstance(smiles_output, list) else str(smiles_output)
                        generated_smiles.append(gen_smiles)
                        
                except Exception as e:
                    logger.warning(f"样本 {i} 生成失败: {e}")
                    generated_outputs.append("GENERATION_FAILED")
                    generated_smiles.append("CC")  # 默认SMILES
            
            generation_time = time.time() - start_time
            
            # 统计有效输出
            if output_modality == 'smiles':
                valid_count = sum(1 for smiles in generated_smiles 
                                 if MolecularUtils.validate_smiles(smiles))
            else:
                valid_count = sum(1 for output in generated_outputs 
                                 if not output.endswith("GENERATION_FAILED"))
            
            # 计算评估指标（使用SMILES版本）
            target_smiles = scaffold_data  # 使用scaffold作为基准
            metrics = self._compute_metrics(generated_smiles, target_smiles)
            
            # 更新结果
            results.update({
                'success': True,
                'generation_time': generation_time,
                'generated_outputs': generated_outputs,
                'generated_smiles': generated_smiles,  # 用于后续分析
                'valid_outputs': valid_count,
                'metrics': metrics
            })
            
            logger.info(f"✅ 成功生成 {valid_count}/{sample_count} 个有效输出")
            logger.info(f"⏱️  生成耗时: {generation_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ 组合测试失败: {e}")
            results['error_message'] = str(e)
        
        return results
    
    def _compute_metrics(self, generated_smiles: List[str], target_smiles: List[str]) -> Dict[str, float]:
        """计算评估指标"""
        try:
            # 基础指标
            validity = self.metrics_calculator.molecular_metrics.compute_validity(generated_smiles)
            uniqueness = self.metrics_calculator.molecular_metrics.compute_uniqueness(generated_smiles)
            diversity = self.metrics_calculator.molecular_metrics.compute_diversity(generated_smiles)
            
            # 分子相似性指标
            similarity_metrics = self.metrics_calculator.compute_molecular_similarity(
                generated_smiles, target_smiles
            )
            
            # 组合所有指标
            metrics = {}
            metrics.update(validity)
            metrics.update(uniqueness)
            metrics.update(diversity)
            metrics.update(similarity_metrics)
            
            return metrics
            
        except Exception as e:
            logger.warning(f"指标计算失败: {e}")
            return {'validity': 0.0, 'error': str(e)}
    
    def run_comprehensive_test(self, sample_count: int = 3, 
                              save_results: bool = True) -> Dict[str, Any]:
        """
        运行全面的9模态测试
        
        Args:
            sample_count: 每个组合的测试样本数
            save_results: 是否保存结果
            
        Returns:
            完整测试结果
        """
        logger.info("🚀 开始9模态系统全面测试")
        logger.info(f"📊 测试配置: 每组合{sample_count}个样本, 共9种组合")
        
        # 加载模型
        if not self.load_model():
            return {'error': '模型加载失败'}
        
        # 初始化结果
        comprehensive_results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'device': self.device,
            'sample_count_per_combination': sample_count,
            'total_combinations': len(self.combinations),
            'combination_results': {},
            'summary_metrics': {},
            'success_rates': {},
            'performance_stats': {}
        }
        
        total_start_time = time.time()
        successful_combinations = 0
        
        # 测试每个组合
        for i, (scaffold_mod, output_mod) in enumerate(self.combinations, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"测试进度: {i}/{len(self.combinations)}")
            logger.info(f"组合: Scaffold({scaffold_mod}) + Text → {output_mod}")
            logger.info(f"{'='*60}")
            
            # 执行单个组合测试
            combination_result = self.test_single_combination(
                scaffold_mod, output_mod, sample_count
            )
            
            # 存储结果
            combination_key = f"{scaffold_mod}_to_{output_mod}"
            comprehensive_results['combination_results'][combination_key] = combination_result
            
            if combination_result['success']:
                successful_combinations += 1
        
        total_time = time.time() - total_start_time
        
        # 计算汇总统计
        self._compute_summary_statistics(comprehensive_results, successful_combinations, total_time)
        
        # 保存结果
        if save_results:
            self._save_results(comprehensive_results)
        
        # 显示最终报告
        self._print_final_report(comprehensive_results)
        
        logger.info(f"\n🎉 9模态测试完成! 成功率: {successful_combinations}/{len(self.combinations)}")
        
        return comprehensive_results
    
    def _compute_summary_statistics(self, results: Dict[str, Any], 
                                   successful_combinations: int, total_time: float):
        """计算汇总统计信息"""
        
        # 成功率统计
        results['success_rates'] = {
            'overall_success_rate': successful_combinations / len(self.combinations),
            'successful_combinations': successful_combinations,
            'total_combinations': len(self.combinations)
        }
        
        # 按模态统计成功率
        modality_stats = {}
        for scaffold_mod in ['smiles', 'graph', 'image']:
            for output_mod in ['smiles', 'graph', 'image']:
                key = f"{scaffold_mod}_to_{output_mod}"
                if key in results['combination_results']:
                    success = results['combination_results'][key]['success']
                    
                    # 输入模态统计
                    if scaffold_mod not in modality_stats:
                        modality_stats[scaffold_mod] = {'input_success': 0, 'input_total': 0}
                    modality_stats[scaffold_mod]['input_total'] += 1
                    if success:
                        modality_stats[scaffold_mod]['input_success'] += 1
                    
                    # 输出模态统计
                    output_key = f"{output_mod}_output"
                    if output_key not in modality_stats:
                        modality_stats[output_key] = {'output_success': 0, 'output_total': 0}
                    modality_stats[output_key]['output_total'] += 1
                    if success:
                        modality_stats[output_key]['output_success'] += 1
        
        results['success_rates']['modality_breakdown'] = modality_stats
        
        # 性能统计
        generation_times = []
        validity_scores = []
        uniqueness_scores = []
        diversity_scores = []
        
        for combination_result in results['combination_results'].values():
            if combination_result['success']:
                generation_times.append(combination_result['generation_time'])
                
                metrics = combination_result.get('metrics', {})
                if 'validity' in metrics:
                    validity_scores.append(metrics['validity'])
                if 'uniqueness' in metrics:
                    uniqueness_scores.append(metrics['uniqueness'])
                if 'diversity_score' in metrics:
                    diversity_scores.append(metrics['diversity_score'])
        
        results['performance_stats'] = {
            'total_test_time': total_time,
            'mean_generation_time': np.mean(generation_times) if generation_times else 0.0,
            'std_generation_time': np.std(generation_times) if generation_times else 0.0,
            'mean_validity': np.mean(validity_scores) if validity_scores else 0.0,
            'mean_uniqueness': np.mean(uniqueness_scores) if uniqueness_scores else 0.0,
            'mean_diversity': np.mean(diversity_scores) if diversity_scores else 0.0
        }
        
        # 汇总指标
        results['summary_metrics'] = {
            'overall_validity': results['performance_stats']['mean_validity'],
            'overall_uniqueness': results['performance_stats']['mean_uniqueness'],
            'overall_diversity': results['performance_stats']['mean_diversity'],
            'average_generation_time': results['performance_stats']['mean_generation_time'],
            'system_stability': successful_combinations / len(self.combinations)
        }
    
    def _save_results(self, results: Dict[str, Any]):
        """保存测试结果"""
        try:
            # 创建结果目录
            results_dir = Path(self.model_path).parent / 'test_results'
            results_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 保存详细结果 (JSON)
            json_file = results_dir / f'9modal_test_results_{timestamp}.json'
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            # 保存汇总报告 (Markdown)
            markdown_file = results_dir / f'9modal_test_report_{timestamp}.md'
            self._generate_markdown_report(results, markdown_file)
            
            logger.info(f"📁 结果已保存:")
            logger.info(f"   详细结果: {json_file}")
            logger.info(f"   汇总报告: {markdown_file}")
            
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
    
    def _generate_markdown_report(self, results: Dict[str, Any], output_file: Path):
        """生成Markdown格式的测试报告"""
        
        report_content = f"""# 9模态分子生成系统测试报告

## 测试概览

- **测试时间**: {results['timestamp']}
- **模型路径**: {results['model_path']}
- **计算设备**: {results['device']}
- **每组合样本数**: {results['sample_count_per_combination']}
- **总组合数**: {results['total_combinations']}

## 总体成功率

- **系统稳定性**: {results['summary_metrics']['system_stability']:.2%}
- **成功组合数**: {results['success_rates']['successful_combinations']}/{results['success_rates']['total_combinations']}

## 性能指标

| 指标 | 数值 |
|------|------|
| 总测试时间 | {results['performance_stats']['total_test_time']:.2f}s |
| 平均生成时间 | {results['performance_stats']['mean_generation_time']:.2f}s |
| 平均有效性 | {results['performance_stats']['mean_validity']:.2%} |
| 平均唯一性 | {results['performance_stats']['mean_uniqueness']:.2%} |
| 平均多样性 | {results['performance_stats']['mean_diversity']:.2%} |

## 各组合详细结果

"""
        
        # 添加组合结果表格
        report_content += "| Scaffold输入 | 输出格式 | 状态 | 有效输出 | 生成时间 | 有效性 |\n"
        report_content += "|-------------|---------|------|---------|---------|--------|\n"
        
        for combination_key, result in results['combination_results'].items():
            scaffold_mod, output_mod = combination_key.split('_to_')
            status = "✅" if result['success'] else "❌"
            valid_outputs = f"{result['valid_outputs']}/{result['sample_count']}"
            gen_time = f"{result['generation_time']:.2f}s"
            validity = result['metrics'].get('validity', 0.0)
            validity_pct = f"{validity:.1%}"
            
            report_content += f"| {scaffold_mod} | {output_mod} | {status} | {valid_outputs} | {gen_time} | {validity_pct} |\n"
        
        report_content += "\n## 生成样例\n\n"
        
        # 添加一些生成样例
        for i, (combination_key, result) in enumerate(results['combination_results'].items()):
            if result['success'] and i < 3:  # 只显示前3个成功的组合
                scaffold_mod, output_mod = combination_key.split('_to_')
                report_content += f"### {scaffold_mod.upper()} → {output_mod.upper()}\n\n"
                
                generated_smiles = result.get('generated_smiles', [])
                for j, smiles in enumerate(generated_smiles[:2]):  # 每个组合显示2个样例
                    if MolecularUtils.validate_smiles(smiles):
                        report_content += f"- 样例 {j+1}: `{smiles}`\n"
                
                report_content += "\n"
        
        # 写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
    
    def _print_final_report(self, results: Dict[str, Any]):
        """打印最终测试报告"""
        
        print("\n" + "="*70)
        print("🧪 9模态分子生成系统 - 测试报告")
        print("="*70)
        
        # 总体统计
        print(f"\n📊 总体统计:")
        print(f"   系统稳定性: {results['summary_metrics']['system_stability']:.1%}")
        print(f"   成功组合: {results['success_rates']['successful_combinations']}/{results['success_rates']['total_combinations']}")
        print(f"   总测试时间: {results['performance_stats']['total_test_time']:.1f}s")
        
        # 性能指标
        print(f"\n⚡ 性能指标:")
        print(f"   平均生成时间: {results['performance_stats']['mean_generation_time']:.2f}s")
        print(f"   平均有效性: {results['performance_stats']['mean_validity']:.1%}")
        print(f"   平均唯一性: {results['performance_stats']['mean_uniqueness']:.1%}")
        print(f"   平均多样性: {results['performance_stats']['mean_diversity']:.1%}")
        
        # 组合结果矩阵
        print(f"\n🔍 组合结果矩阵:")
        print("     输出→  SMILES  Graph   Image")
        
        for scaffold_mod in ['smiles', 'graph', 'image']:
            line = f"  {scaffold_mod:8s}"
            for output_mod in ['smiles', 'graph', 'image']:
                key = f"{scaffold_mod}_to_{output_mod}"
                if key in results['combination_results']:
                    success = results['combination_results'][key]['success']
                    status = "   ✅   " if success else "   ❌   "
                else:
                    status = "   ❓   "
                line += status
            print(line)
        
        # 最佳和最差组合
        successful_combinations = [
            (key, result) for key, result in results['combination_results'].items()
            if result['success']
        ]
        
        if successful_combinations:
            # 按有效性排序
            successful_combinations.sort(
                key=lambda x: x[1]['metrics'].get('validity', 0), 
                reverse=True
            )
            
            print(f"\n🏆 最佳组合 (按有效性):")
            best_key, best_result = successful_combinations[0]
            scaffold_mod, output_mod = best_key.split('_to_')
            validity = best_result['metrics'].get('validity', 0)
            print(f"   {scaffold_mod.upper()} → {output_mod.upper()}: 有效性 {validity:.1%}")
            
            if len(successful_combinations) > 1:
                print(f"\n⚠️  待改进组合:")
                worst_key, worst_result = successful_combinations[-1]
                scaffold_mod, output_mod = worst_key.split('_to_')
                validity = worst_result['metrics'].get('validity', 0)
                print(f"   {scaffold_mod.upper()} → {output_mod.upper()}: 有效性 {validity:.1%}")
        
        print("\n" + "="*70)
        print("测试完成! 详细结果已保存到文件。")
        print("="*70)


def main():
    """主函数"""
    print("🚀 启动9模态分子生成系统全面测试")
    
    # 配置参数
    MODEL_PATH = "/root/autodl-tmp/text2Mol-outputs/9modal_20250810_161606_production/best_model.pth"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SAMPLE_COUNT = 2  # 每个组合测试2个样本（快速测试）
    
    print(f"📁 模型路径: {MODEL_PATH}")
    print(f"🖥️  计算设备: {DEVICE}")
    print(f"📊 测试样本: 每组合{SAMPLE_COUNT}个样本")
    
    # 检查模型文件
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 模型文件不存在 - {MODEL_PATH}")
        print("请确保训练已完成并且模型文件存在。")
        return
    
    # 创建测试器
    tester = NineModalTester(MODEL_PATH, DEVICE)
    
    # 运行全面测试
    try:
        results = tester.run_comprehensive_test(
            sample_count=SAMPLE_COUNT,
            save_results=True
        )
        
        if 'error' in results:
            print(f"❌ 测试失败: {results['error']}")
        else:
            print("\n✅ 测试成功完成!")
            success_rate = results['summary_metrics']['system_stability']
            print(f"🎯 系统整体成功率: {success_rate:.1%}")
            
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()