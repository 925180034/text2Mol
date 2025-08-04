"""
多模态分子生成模型训练脚本
完整的训练流程，包括数据加载、模型训练、验证和保存
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from scaffold_mol_gen.models.end2end_model import End2EndMolecularGenerator
from scaffold_mol_gen.data.multimodal_dataset import create_data_loaders
from scaffold_mol_gen.training.metrics import GenerationMetrics
from scaffold_mol_gen.utils.mol_utils import MolecularUtils

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiModalTrainer:
    """多模态分子生成模型训练器"""
    
    def __init__(self,
                 model: End2EndMolecularGenerator,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict[str, Any],
                 device: str = 'cuda'):
        """
        Args:
            model: 端到端模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            config: 训练配置
            device: 设备
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # 训练参数
        self.num_epochs = config.get('num_epochs', 10)
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.weight_decay = config.get('weight_decay', 1e-5)
        self.warmup_steps = config.get('warmup_steps', 1000)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.save_interval = config.get('save_interval', 1000)
        self.eval_interval = config.get('eval_interval', 500)
        self.log_interval = config.get('log_interval', 100)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            epochs=self.num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1
        )
        
        # 评价指标
        self.metrics = GenerationMetrics()
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # 输出目录
        self.output_dir = Path(config.get('output_dir', 'outputs/training'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(self.output_dir / 'tensorboard')
        
        # 保存配置
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"训练器初始化完成")
        logger.info(f"  - 设备: {device}")
        logger.info(f"  - 训练批次: {len(train_loader)}")
        logger.info(f"  - 验证批次: {len(val_loader)}")
        logger.info(f"  - 学习率: {self.learning_rate}")
        logger.info(f"  - 输出目录: {self.output_dir}")
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch + 1}/{self.num_epochs}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # 前向传播
            loss = self.train_step(batch)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
            
            # 更新参数
            self.optimizer.step()
            self.scheduler.step()
            
            # 统计
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # 日志记录
            if self.global_step % self.log_interval == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/learning_rate', 
                                     self.optimizer.param_groups[0]['lr'], 
                                     self.global_step)
            
            # 验证
            if self.global_step % self.eval_interval == 0:
                val_metrics = self.validate()
                self.model.train()  # 切回训练模式
                
                # 记录验证指标
                for key, value in val_metrics.items():
                    self.writer.add_scalar(f'val/{key}', value, self.global_step)
                
                logger.info(f"Step {self.global_step} - Val Loss: {val_metrics.get('loss', 0):.4f}")
            
            # 保存检查点
            if self.global_step % self.save_interval == 0:
                self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'loss': avg_loss}
    
    def train_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        """单步训练"""
        # 移动数据到设备
        scaffold_data = batch['scaffold_data']
        text_data = batch['text_data']
        target_smiles = batch['target_smiles']
        scaffold_modality = batch['scaffold_modality']
        
        # 计算损失
        loss = self.model.compute_loss(
            scaffold_data=scaffold_data,
            text_data=text_data,
            target_smiles=target_smiles,
            scaffold_modality=scaffold_modality
        )
        
        return loss
    
    def validate(self) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        generated_smiles = []
        target_smiles_list = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                # 计算损失
                loss = self.train_step(batch)
                total_loss += loss.item()
                num_batches += 1
                
                # 生成样本（少量用于质量评估）
                if len(generated_smiles) < 100:  # 只评估前100个样本
                    try:
                        scaffold_data = batch['scaffold_data'][:1]  # 只取第一个
                        text_data = batch['text_data'][:1]
                        target = batch['target_smiles'][:1]
                        scaffold_modality = batch['scaffold_modality']
                        
                        generated = self.model.generate(
                            scaffold_data=scaffold_data,
                            text_data=text_data,
                            scaffold_modality=scaffold_modality,
                            num_beams=3,
                            max_length=64
                        )
                        
                        generated_smiles.extend(generated)
                        target_smiles_list.extend(target)
                        
                    except Exception as e:
                        logger.warning(f"生成过程出错: {e}")
        
        # 计算验证指标
        val_metrics = {'loss': total_loss / num_batches if num_batches > 0 else 0.0}
        
        # 计算分子质量指标
        if generated_smiles and target_smiles_list:
            try:
                quality_metrics = self.metrics.compute_metrics(
                    generated_smiles=generated_smiles,
                    reference_smiles=target_smiles_list
                )
                val_metrics.update(quality_metrics)
                
                logger.info("验证指标:")
                for key, value in quality_metrics.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"  {key}: {value:.4f}")
                
            except Exception as e:
                logger.warning(f"质量指标计算失败: {e}")
        
        return val_metrics
    
    def train(self):
        """完整训练流程"""
        logger.info("开始训练...")
        logger.info(f"训练 {self.num_epochs} epochs")
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            logger.info(f"\n=== Epoch {epoch + 1}/{self.num_epochs} ===")
            
            # 训练一个epoch
            train_metrics = self.train_epoch()
            
            # 记录epoch指标
            self.writer.add_scalar('train/epoch_loss', train_metrics['loss'], epoch)
            
            # Epoch结束时验证
            val_metrics = self.validate()
            
            # 记录验证指标
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'val_epoch/{key}', value, epoch)
            
            # 保存最佳模型
            val_loss = val_metrics.get('loss', float('inf'))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pt', is_best=True)
                logger.info(f"✅ 保存最佳模型 (Val Loss: {val_loss:.4f})")
            
            # 保存epoch检查点
            self.save_checkpoint(f'epoch_{epoch + 1}.pt')
            
            logger.info(f"Epoch {epoch + 1} - Train Loss: {train_metrics['loss']:.4f}, "
                       f"Val Loss: {val_loss:.4f}")
        
        total_time = time.time() - start_time
        logger.info(f"\n✅ 训练完成! 总用时: {total_time/3600:.2f} 小时")
        
        # 最终保存
        self.save_checkpoint('final_model.pt')
        self.writer.close()
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        checkpoint_path = self.output_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            # 额外保存到easy-to-find位置
            best_path = self.output_dir / 'model_best.pt'
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"从检查点恢复: epoch {self.epoch}, step {self.global_step}")


def create_training_config() -> Dict[str, Any]:
    """创建默认训练配置"""
    return {
        'num_epochs': 5,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'warmup_steps': 500,
        'max_grad_norm': 1.0,
        'batch_size': 8,
        'num_workers': 4,
        'save_interval': 1000,
        'eval_interval': 500,
        'log_interval': 50,
        'output_dir': f'outputs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'scaffold_modality': 'smiles',
        'max_text_length': 128,
        'max_smiles_length': 128,
        'filter_invalid': True,
        'augment_data': False
    }


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='训练多模态分子生成模型')
    parser.add_argument('--train-data', type=str, default='Datasets/train.csv',
                       help='训练数据CSV文件')
    parser.add_argument('--val-data', type=str, default='Datasets/validation.csv',
                       help='验证数据CSV文件')
    parser.add_argument('--test-data', type=str, default='Datasets/test.csv',
                       help='测试数据CSV文件')
    parser.add_argument('--output-dir', type=str, 
                       default=f'outputs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                       help='输出目录')
    parser.add_argument('--batch-size', type=int, default=8, help='批大小')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--scaffold-modality', type=str, default='smiles',
                       choices=['smiles', 'graph', 'image'], help='Scaffold模态')
    parser.add_argument('--resume', type=str, help='从检查点恢复训练')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    
    args = parser.parse_args()
    
    # 检查数据文件
    if not os.path.exists(args.train_data):
        logger.error(f"训练数据文件不存在: {args.train_data}")
        return
    
    if not os.path.exists(args.val_data):
        logger.error(f"验证数据文件不存在: {args.val_data}")
        return
    
    # 创建配置
    config = create_training_config()
    config.update({
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'learning_rate': args.lr,
        'output_dir': args.output_dir,
        'scaffold_modality': args.scaffold_modality
    })
    
    logger.info(f"训练配置: {json.dumps(config, indent=2)}")
    
    # 创建数据加载器
    try:
        data_loaders = create_data_loaders(
            train_csv=args.train_data,
            val_csv=args.val_data,
            test_csv=args.test_data if os.path.exists(args.test_data) else None,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            scaffold_modality=config['scaffold_modality'],
            max_text_length=config['max_text_length'],
            max_smiles_length=config['max_smiles_length'],
            filter_invalid=config['filter_invalid'],
            augment_data=config['augment_data']
        )
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        return
    
    # 创建模型
    try:
        logger.info("创建端到端模型...")
        model = End2EndMolecularGenerator(
            hidden_size=768,
            molt5_path="/root/autodl-tmp/text2Mol-models/MolT5-Large-Caption2SMILES",
            use_scibert=False,
            freeze_encoders=True,
            freeze_molt5=True,
            fusion_type='both',
            device=args.device
        )
        logger.info(f"✅ 模型创建成功")
    except Exception as e:
        logger.error(f"模型创建失败: {e}")
        return
    
    # 创建训练器
    trainer = MultiModalTrainer(
        model=model,
        train_loader=data_loaders['train'],
        val_loader=data_loaders['val'],
        config=config,
        device=args.device
    )
    
    # 恢复训练（如果指定）
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    try:
        trainer.train()
        logger.info("🎉 训练成功完成！")
    except KeyboardInterrupt:
        logger.info("⚠️ 训练被用户中断")
        trainer.save_checkpoint('interrupted_checkpoint.pt')
    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()