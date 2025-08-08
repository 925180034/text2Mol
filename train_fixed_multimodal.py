#!/usr/bin/env python3
"""
修复的多模态分子生成模型训练脚本
解决tokenizer范围错误和无效生成问题
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import pandas as pd

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from transformers import T5ForConditionalGeneration, T5Tokenizer
from scaffold_mol_gen.models.end2end_model import End2EndMolecularGenerator
from scaffold_mol_gen.training.metrics import GenerationMetrics
from scaffold_mol_gen.utils.mol_utils import MolecularUtils
from scaffold_mol_gen.data.multimodal_preprocessor import MultiModalPreprocessor
from rdkit import Chem

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FixedMultiModalDataset(Dataset):
    """修复的多模态数据集，确保tokenizer兼容性"""
    
    def __init__(self, csv_file: str, 
                 molt5_tokenizer: T5Tokenizer,
                 scaffold_modality: str = 'smiles',
                 max_text_length: int = 128,
                 max_smiles_length: int = 128,
                 filter_invalid: bool = True):
        """
        Args:
            csv_file: CSV数据文件路径
            molt5_tokenizer: MolT5 tokenizer
            scaffold_modality: scaffold模态类型
            max_text_length: 最大文本长度
            max_smiles_length: 最大SMILES长度
            filter_invalid: 是否过滤无效数据
        """
        self.tokenizer = molt5_tokenizer
        self.scaffold_modality = scaffold_modality
        self.max_text_length = max_text_length
        self.max_smiles_length = max_smiles_length
        self.preprocessor = MultiModalPreprocessor()
        
        # 加载和处理数据
        logger.info(f"加载数据集: {csv_file}")
        df = pd.read_csv(csv_file)
        
        # 数据清洗和验证
        self.data = []
        valid_count = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理数据"):
            scaffold = str(row.get('scaffold', row.get('SMILES', ''))).strip()
            text = str(row.get('description', row.get('text', ''))).strip()
            target_smiles = str(row.get('SMILES', row.get('target', ''))).strip()
            
            # 验证数据有效性
            if filter_invalid:
                # 验证目标SMILES
                if not target_smiles or target_smiles in ['nan', 'None']:
                    continue
                
                mol = Chem.MolFromSmiles(target_smiles)
                if mol is None:
                    continue
                
                # 规范化SMILES
                target_smiles = Chem.MolToSmiles(mol, canonical=True)
                
                # 验证scaffold（如果是SMILES模态）
                if scaffold_modality == 'smiles':
                    if not scaffold or scaffold in ['nan', 'None']:
                        scaffold = ""  # 空scaffold
                    elif Chem.MolFromSmiles(scaffold) is None:
                        continue
                
                # 验证文本描述
                if not text or text in ['nan', 'None']:
                    text = ""  # 空文本描述
            
            # 截断长度
            text = text[:max_text_length*2]  # 预留tokenization空间
            target_smiles = target_smiles[:max_smiles_length*2]
            
            self.data.append({
                'scaffold': scaffold,
                'text': text,
                'target_smiles': target_smiles
            })
            valid_count += 1
        
        logger.info(f"数据集加载完成: {len(df)} -> {valid_count} 有效样本")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # 构建输入文本（MolT5格式）
        scaffold_text = f"Scaffold: {sample['scaffold']}" if sample['scaffold'] else "Scaffold: <empty>"
        description_text = f"Description: {sample['text']}" if sample['text'] else "Description: <empty>"
        input_text = f"{scaffold_text} {description_text}"
        
        # Tokenize输入
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_text_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Tokenize目标
        target_encoding = self.tokenizer(
            sample['target_smiles'],
            max_length=self.max_smiles_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(0),
            'attention_mask': input_encoding['attention_mask'].squeeze(0),
            'labels': target_encoding['input_ids'].squeeze(0),
            'target_attention_mask': target_encoding['attention_mask'].squeeze(0),
            'scaffold_data': sample['scaffold'],
            'text_data': sample['text'],
            'target_smiles': sample['target_smiles'],
            'scaffold_modality': self.scaffold_modality
        }


class ConstrainedMultiModalTrainer:
    """约束的多模态训练器，防止生成超出词汇表的token"""
    
    def __init__(self,
                 model: End2EndMolecularGenerator,
                 train_dataset: FixedMultiModalDataset,
                 val_dataset: FixedMultiModalDataset,
                 config: Dict[str, Any],
                 device: str = 'cuda'):
        """
        Args:
            model: 端到端模型
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            config: 训练配置
            device: 设备
        """
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = device
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.get('batch_size', 8),
            shuffle=True,
            num_workers=config.get('num_workers', 4),
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.get('batch_size', 8),
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            drop_last=False
        )
        
        # 训练参数
        self.num_epochs = config.get('num_epochs', 5)
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.weight_decay = config.get('weight_decay', 1e-5)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.save_interval = config.get('save_interval', 1000)
        self.eval_interval = config.get('eval_interval', 500)
        self.log_interval = config.get('log_interval', 100)
        
        # Tokenizer约束
        self.tokenizer = train_dataset.tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        logger.info(f"Tokenizer词汇表大小: {self.vocab_size}")
        
        # 优化器（只训练非冻结参数）
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            epochs=self.num_epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.1
        )
        
        # 评价指标
        self.metrics = GenerationMetrics()
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # 输出目录
        self.output_dir = Path(config.get('output_dir', 'outputs/fixed_training'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(self.output_dir / 'tensorboard')
        
        # 保存配置
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"训练器初始化完成")
        logger.info(f"  - 设备: {device}")
        logger.info(f"  - 训练样本: {len(train_dataset)}")
        logger.info(f"  - 验证样本: {len(val_dataset)}")
        logger.info(f"  - 批大小: {config.get('batch_size', 8)}")
        logger.info(f"  - 学习率: {self.learning_rate}")
        logger.info(f"  - 可训练参数: {sum(p.numel() for p in trainable_params):,}")
    
    def compute_constrained_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """计算约束损失，防止生成无效token"""
        # 移动数据到设备
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # 前向传播（使用正确的模型属性）
        outputs = self.model.generator.molt5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # 基础损失
        loss = outputs.loss
        
        # 约束项：防止生成超出词汇表的token
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
            
            # 约束logits到有效词汇表范围
            # 对超出词汇表的位置施加大的负值
            if logits.size(-1) > self.vocab_size:
                invalid_mask = torch.zeros_like(logits)
                invalid_mask[:, :, self.vocab_size:] = -float('inf')
                logits = logits + invalid_mask
                
                # 重新计算损失
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                if labels is not None:
                    # Shift so that tokens < n predict n
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                                  shift_labels.view(-1))
        
        return loss
    
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
            # 计算损失
            loss = self.compute_constrained_loss(batch)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
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
                loss = self.compute_constrained_loss(batch)
                total_loss += loss.item()
                num_batches += 1
                
                # 生成样本（少量用于质量评估）
                if len(generated_smiles) < 50:  # 限制验证样本数量
                    try:
                        # 使用约束生成
                        input_ids = batch['input_ids'][:2].to(self.device)  # 只取2个样本
                        attention_mask = batch['attention_mask'][:2].to(self.device)
                        
                        generated_ids = self.model.generator.molt5.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_length=64,
                            num_beams=3,
                            temperature=0.9,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            forced_eos_token_id=self.tokenizer.eos_token_id
                        )
                        
                        # 约束生成的token ID到有效范围
                        generated_ids = torch.clamp(generated_ids, 0, self.vocab_size - 1)
                        
                        batch_generated = self.tokenizer.batch_decode(
                            generated_ids,
                            skip_special_tokens=True
                        )
                        
                        batch_targets = batch['target_smiles'][:2]
                        
                        generated_smiles.extend(batch_generated)
                        target_smiles_list.extend(batch_targets)
                        
                    except Exception as e:
                        logger.warning(f"验证生成过程出错: {e}")
        
        # 计算验证指标
        val_metrics = {'loss': total_loss / num_batches if num_batches > 0 else 0.0}
        
        # 计算分子质量指标
        if generated_smiles and target_smiles_list:
            try:
                # 验证生成的SMILES
                valid_generated = []
                valid_targets = []
                for gen, target in zip(generated_smiles, target_smiles_list):
                    if gen and target:
                        # 验证生成的SMILES
                        mol = Chem.MolFromSmiles(gen)
                        if mol is not None:
                            valid_generated.append(gen)
                            valid_targets.append(target)
                
                if valid_generated:
                    quality_metrics = self.metrics.compute_metrics(
                        generated_smiles=valid_generated,
                        reference_smiles=valid_targets
                    )
                    val_metrics.update(quality_metrics)
                    
                    logger.info("验证指标:")
                    logger.info(f"  生成样本: {len(generated_smiles)}, 有效: {len(valid_generated)}")
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
            'config': self.config,
            'vocab_size': self.vocab_size
        }
        
        checkpoint_path = self.output_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            # 额外保存到easy-to-find位置
            best_path = self.output_dir / 'model_best.pt'
            torch.save(checkpoint, best_path)


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='修复的多模态分子生成模型训练')
    parser.add_argument('--train-data', type=str, default='Datasets/train.csv',
                       help='训练数据CSV文件')
    parser.add_argument('--val-data', type=str, default='Datasets/validation.csv',
                       help='验证数据CSV文件')
    parser.add_argument('--output-dir', type=str, 
                       default=f'outputs/fixed_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                       help='输出目录')
    parser.add_argument('--batch-size', type=int, default=4, help='批大小')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--lr', type=float, default=5e-5, help='学习率')
    parser.add_argument('--scaffold-modality', type=str, default='smiles',
                       choices=['smiles', 'graph', 'image'], help='Scaffold模态')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--sample-size', type=int, default=1000, help='训练样本数限制')
    
    args = parser.parse_args()
    
    # 检查数据文件
    if not os.path.exists(args.train_data):
        logger.error(f"训练数据文件不存在: {args.train_data}")
        return
    
    if not os.path.exists(args.val_data):
        logger.error(f"验证数据文件不存在: {args.val_data}")
        return
    
    # 创建配置
    config = {
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': 1e-5,
        'max_grad_norm': 1.0,
        'num_workers': 2,
        'save_interval': 500,
        'eval_interval': 200,
        'log_interval': 50,
        'output_dir': args.output_dir,
        'scaffold_modality': args.scaffold_modality,
        'max_text_length': 128,
        'max_smiles_length': 128,
        'filter_invalid': True
    }
    
    logger.info(f"训练配置: {json.dumps(config, indent=2)}")
    
    # 初始化tokenizer
    molt5_path = "/root/autodl-tmp/text2Mol-models/MolT5-Large-Caption2SMILES"
    tokenizer = T5Tokenizer.from_pretrained(molt5_path)
    
    # 创建数据集
    try:
        logger.info("创建训练数据集...")
        train_dataset = FixedMultiModalDataset(
            csv_file=args.train_data,
            molt5_tokenizer=tokenizer,
            scaffold_modality=config['scaffold_modality'],
            max_text_length=config['max_text_length'],
            max_smiles_length=config['max_smiles_length'],
            filter_invalid=config['filter_invalid']
        )
        
        # 限制样本数量
        if args.sample_size and len(train_dataset.data) > args.sample_size:
            train_dataset.data = train_dataset.data[:args.sample_size]
            logger.info(f"限制训练样本数为: {len(train_dataset.data)}")
        
        logger.info("创建验证数据集...")
        val_dataset = FixedMultiModalDataset(
            csv_file=args.val_data,
            molt5_tokenizer=tokenizer,
            scaffold_modality=config['scaffold_modality'],
            max_text_length=config['max_text_length'],
            max_smiles_length=config['max_smiles_length'],
            filter_invalid=config['filter_invalid']
        )
        
        # 限制验证样本
        if len(val_dataset.data) > 200:
            val_dataset.data = val_dataset.data[:200]
        
        logger.info(f"数据集创建完成: train={len(train_dataset)}, val={len(val_dataset)}")
        
    except Exception as e:
        logger.error(f"数据集创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建模型
    try:
        logger.info("创建端到端模型...")
        model = End2EndMolecularGenerator(
            hidden_size=768,
            molt5_path=molt5_path,
            use_scibert=False,
            freeze_encoders=True,
            freeze_molt5=False,  # 解冻MolT5进行微调
            fusion_type='both',
            device=args.device
        )
        logger.info(f"✅ 模型创建成功")
        
        # 打印可训练参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"总参数: {total_params:,}, 可训练: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        
    except Exception as e:
        logger.error(f"模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建训练器
    try:
        trainer = ConstrainedMultiModalTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config,
            device=args.device
        )
        
        # 开始训练
        logger.info("🚀 开始修复的多模态训练...")
        trainer.train()
        logger.info("🎉 训练成功完成！")
        
    except KeyboardInterrupt:
        logger.info("⚠️ 训练被用户中断")
        if 'trainer' in locals():
            trainer.save_checkpoint('interrupted_checkpoint.pt')
    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()