#!/usr/bin/env python3
"""
快速启动完整数据集实验
一键运行所有核心实验功能
"""

import argparse
import logging
import time
from pathlib import Path
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_quick_validation():
    """运行快速验证测试"""
    logger.info("🧪 开始快速验证测试...")
    
    try:
        from optimized_prompting import OptimizedMolecularPrompting
        from extended_input_output import ExtendedModalitySystem
        
        # 测试优化prompting
        logger.info("1. 测试优化Prompting系统...")
        prompting = OptimizedMolecularPrompting()
        
        test_molecules = ['water', 'ethanol', 'benzene', 'aspirin']
        results = []
        
        for mol in test_molecules:
            result = prompting.generate_smiles(mol, num_candidates=3)
            if result['generation_success'] and result['best_candidate']['is_valid']:
                smiles = result['best_candidate']['smiles']
                results.append((mol, smiles))
                logger.info(f"  ✅ {mol} → {smiles}")
            else:
                logger.warning(f"  ❌ {mol} → 生成失败")
        
        # 测试扩展模态
        logger.info("2. 测试扩展模态系统...")
        modality_system = ExtendedModalitySystem()
        
        for mol, smiles in results:
            # 测试SMILES→Properties
            props = modality_system.smiles_to_properties(smiles)
            if props['success']:
                p = props['properties']
                logger.info(f"  📊 {smiles}: MW={p['molecular_weight']:.1f}, LogP={p['logp']:.2f}")
            
            # 测试多模态融合
            fusion = modality_system.multi_modal_fusion(mol, smiles)
            if fusion['success']:
                logger.info(f"  🔄 {mol}+{smiles}: 融合成功 ({fusion['fused_embedding'].shape})")
        
        logger.info("✅ 快速验证完成!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 快速验证失败: {e}")
        return False

def run_short_training():
    """运行短期训练实验 (5个epoch)"""
    logger.info("🚀 开始短期训练实验...")
    
    try:
        from enhanced_training_pipeline import TrainingConfig, EnhancedTrainer, MolecularDataset
        from transformers import T5Tokenizer
        
        # 配置短期训练
        config = TrainingConfig()
        config.num_epochs = 5
        config.batch_size = 4  # 内存友好
        config.gradient_accumulation_steps = 4
        config.eval_steps = 200
        config.save_steps = 500
        config.learning_rate = 5e-5
        
        logger.info(f"训练配置: {config.num_epochs} epochs, batch_size={config.batch_size}")
        
        # 加载tokenizer
        tokenizer = T5Tokenizer.from_pretrained(config.model_path)
        
        # 创建数据集 (限制样本数量)
        train_dataset = MolecularDataset(config.train_data, tokenizer, config, is_training=True)
        val_dataset = MolecularDataset(config.val_data, tokenizer, config, is_training=False)
        
        # 限制训练数据量 (快速实验)
        if len(train_dataset) > 5000:
            train_dataset.data = train_dataset.data[:5000]
            train_dataset.difficulty_scores = train_dataset.difficulty_scores[:5000]
            train_dataset._initialize_curriculum()
        
        logger.info(f"训练数据: {len(train_dataset)} 样本")
        logger.info(f"验证数据: {len(val_dataset)} 样本")
        
        # 创建训练器
        trainer = EnhancedTrainer(config)
        
        # 开始训练
        start_time = time.time()
        training_stats = trainer.train(train_dataset, val_dataset)
        training_time = time.time() - start_time
        
        logger.info(f"✅ 短期训练完成! 用时: {training_time/60:.1f} 分钟")
        logger.info(f"训练步数: {len(training_stats['losses'])}")
        logger.info(f"最佳验证损失: {min(training_stats['losses']):.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 短期训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_generation_test(model_path: str = None):
    """运行分子生成测试"""
    logger.info("🧬 开始分子生成测试...")
    
    if model_path is None:
        model_path = "/root/autodl-tmp/text2Mol-outputs/enhanced_training/best_model"
    
    try:
        from optimized_prompting import OptimizedMolecularPrompting
        
        # 使用优化的prompting系统
        prompting = OptimizedMolecularPrompting()
        
        # 测试分子描述
        test_descriptions = [
            "water molecule",
            "simple alcohol with two carbons", 
            "aromatic compound with six carbons",
            "carboxylic acid with three carbons",
            "cyclic alkane with five carbons",
            "anti-inflammatory drug",
            "painkiller medication",
            "glucose sugar molecule"
        ]
        
        results = []
        valid_count = 0
        
        logger.info(f"生成 {len(test_descriptions)} 个分子...")
        
        for i, desc in enumerate(test_descriptions):
            result = prompting.generate_smiles(desc, num_candidates=5, use_ensemble=True)
            
            if result['best_candidate'] and result['best_candidate']['is_valid']:
                best = result['best_candidate']
                results.append({
                    'description': desc,
                    'smiles': best['smiles'],
                    'molecular_weight': best.get('molecular_weight', 0),
                    'logp': best.get('logp', 0),
                    'template_category': result['template_category']
                })
                valid_count += 1
                logger.info(f"  {i+1}. {desc}")
                logger.info(f"     → {best['smiles']}")
                logger.info(f"     → MW: {best.get('molecular_weight', 0):.1f}, LogP: {best.get('logp', 0):.2f}")
            else:
                logger.warning(f"  {i+1}. {desc} → 生成失败")
        
        success_rate = valid_count / len(test_descriptions) * 100
        logger.info(f"✅ 生成测试完成!")
        logger.info(f"成功率: {success_rate:.1f}% ({valid_count}/{len(test_descriptions)})")
        
        # 保存结果
        output_file = "generation_test_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"结果已保存到: {output_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 分子生成测试失败: {e}")
        return False

def run_full_experiment():
    """运行完整数据集实验"""
    logger.info("🎯 开始完整数据集实验...")
    
    try:
        from enhanced_training_pipeline import run_enhanced_training
        
        logger.info("使用增强训练管道进行完整训练...")
        logger.info("预计训练时间: 8-12小时")
        logger.info("训练数据: 26,402 分子样本")
        logger.info("验证数据: 3,299 分子样本")
        
        # 运行增强训练
        training_stats = run_enhanced_training()
        
        if training_stats:
            logger.info("✅ 完整实验成功完成!")
            return True
        else:
            logger.error("❌ 完整实验失败")
            return False
            
    except Exception as e:
        logger.error(f"❌ 完整实验失败: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="快速启动分子生成实验")
    parser.add_argument('--mode', type=str, default='quick', 
                       choices=['quick', 'short', 'generate', 'full'],
                       help='实验模式: quick(快速验证), short(短期训练), generate(生成测试), full(完整训练)')
    parser.add_argument('--model-path', type=str, default=None,
                       help='模型路径 (用于生成测试)')
    
    args = parser.parse_args()
    
    logger.info("🧬 分子生成系统实验启动器")
    logger.info("=" * 50)
    
    success = False
    
    if args.mode == 'quick':
        logger.info("模式: 快速验证 (预计 5-10 分钟)")
        success = run_quick_validation()
        
    elif args.mode == 'short':
        logger.info("模式: 短期训练 (预计 2-4 小时)")
        success = run_short_training()
        
    elif args.mode == 'generate':
        logger.info("模式: 分子生成测试 (预计 10-15 分钟)")
        success = run_generation_test(args.model_path)
        
    elif args.mode == 'full':
        logger.info("模式: 完整数据集训练 (预计 8-12 小时)")
        success = run_full_experiment()
    
    if success:
        logger.info("🎉 实验成功完成!")
        if args.mode == 'quick':
            logger.info("💡 建议下一步: 运行短期训练 python quick_start_experiment.py --mode short")
        elif args.mode == 'short':
            logger.info("💡 建议下一步: 运行生成测试 python quick_start_experiment.py --mode generate")
        elif args.mode == 'generate':
            logger.info("💡 建议下一步: 运行完整训练 python quick_start_experiment.py --mode full")
    else:
        logger.error("❌ 实验失败，请检查错误信息")

if __name__ == '__main__':
    main()