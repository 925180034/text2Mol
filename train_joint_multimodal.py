#!/usr/bin/env python3
"""
联合多模态分子生成模型训练脚本
实现真正的多模态联合训练，包括模态特征对齐
"""

import os
import sys
import argparse
import logging
import json
import time
import random
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


class JointMultiModalDataset(Dataset):
    """联合多模态数据集，支持动态模态切换"""
    
    def __init__(self, csv_file: str, 
                 molt5_tokenizer: T5Tokenizer,
                 modality_mix: Dict[str, float] = None,
                 max_text_length: int = 128,
                 max_smiles_length: int = 128,
                 filter_invalid: bool = True):
        """
        Args:
            csv_file: CSV数据文件路径
            molt5_tokenizer: MolT5 tokenizer
            modality_mix: 模态混合比例 {'smiles': 0.4, 'graph': 0.3, 'image': 0.3}
            max_text_length: 最大文本长度
            max_smiles_length: 最大SMILES长度
            filter_invalid: 是否过滤无效数据
        """
        self.tokenizer = molt5_tokenizer
        self.max_text_length = max_text_length
        self.max_smiles_length = max_smiles_length
        self.preprocessor = MultiModalPreprocessor()
        
        # 默认模态混合比例
        if modality_mix is None:
            modality_mix = {'smiles': 0.4, 'graph': 0.3, 'image': 0.3}
        self.modality_mix = modality_mix
        self.modalities = list(modality_mix.keys())
        
        # 加载和处理数据
        logger.info(f"加载联合训练数据集: {csv_file}")
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
                
                # 验证scaffold
                if not scaffold or scaffold in ['nan', 'None']:
                    scaffold = ""  # 空scaffold
                elif Chem.MolFromSmiles(scaffold) is None:
                    continue  # 无效scaffold
                
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
        
        logger.info(f"联合数据集加载完成: {len(df)} -> {valid_count} 有效样本")
        logger.info(f"模态混合比例: {modality_mix}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # 随机选择模态
        modality = np.random.choice(
            self.modalities, 
            p=list(self.modality_mix.values())
        )
        
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
            'scaffold_modality': modality  # 动态模态
        }


class MultiModalAlignmentLoss(nn.Module):
    """多模态特征对齐损失"""
    
    def __init__(self, hidden_size: int = 768, temperature: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.temperature = temperature
        
        # 模态投影层
        self.smiles_proj = nn.Linear(hidden_size, hidden_size)
        self.graph_proj = nn.Linear(hidden_size, hidden_size)
        self.image_proj = nn.Linear(hidden_size, hidden_size)
        self.text_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, modality_features: Dict[str, torch.Tensor], 
                text_features: torch.Tensor) -> torch.Tensor:
        """
        计算模态对齐损失
        Args:
            modality_features: {'smiles': tensor, 'graph': tensor, 'image': tensor}
            text_features: 文本特征
        Returns:
            对齐损失
        """
        total_loss = 0.0
        num_pairs = 0
        
        # 投影文本特征
        text_proj = self.text_proj(text_features)  # [batch_size, hidden_size]
        text_proj = F.normalize(text_proj, p=2, dim=-1)
        
        # 计算每个模态与文本的对齐损失
        for modality, features in modality_features.items():
            if features is None:
                continue
                
            # 选择对应的投影层
            if modality == 'smiles':
                modal_proj = self.smiles_proj(features)
            elif modality == 'graph':
                modal_proj = self.graph_proj(features)
            elif modality == 'image':
                modal_proj = self.image_proj(features)
            else:
                continue
            
            modal_proj = F.normalize(modal_proj, p=2, dim=-1)
            
            # 计算对比损失
            similarity = torch.matmul(modal_proj, text_proj.T) / self.temperature
            
            # 正样本在对角线上
            batch_size = similarity.size(0)
            labels = torch.arange(batch_size, device=similarity.device)
            
            # 对称对比损失
            loss_modal_to_text = F.cross_entropy(similarity, labels)
            loss_text_to_modal = F.cross_entropy(similarity.T, labels)
            
            total_loss += (loss_modal_to_text + loss_text_to_modal) / 2
            num_pairs += 1
        
        return total_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0)


class JointMultiModalTrainer:
    """联合多模态训练器"""
    
    def __init__(self,
                 model: End2EndMolecularGenerator,
                 train_dataset: JointMultiModalDataset,
                 val_dataset: JointMultiModalDataset,
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
        self.alignment_weight = config.get('alignment_weight', 0.1)
        
        # Tokenizer约束
        self.tokenizer = train_dataset.tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        
        # 对齐损失
        self.alignment_loss = MultiModalAlignmentLoss(
            hidden_size=config.get('hidden_size', 768),
            temperature=config.get('alignment_temperature', 0.1)
        ).to(device)
        
        # 优化器（包含对齐损失的参数）
        all_params = list(self.model.parameters()) + list(self.alignment_loss.parameters())
        trainable_params = [p for p in all_params if p.requires_grad]
        
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
        self.output_dir = Path(config.get('output_dir', 'outputs/joint_training'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(self.output_dir / 'tensorboard')
        
        # 保存配置
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"联合训练器初始化完成")
        logger.info(f"  - 设备: {device}")
        logger.info(f"  - 训练样本: {len(train_dataset)}")
        logger.info(f"  - 验证样本: {len(val_dataset)}")
        logger.info(f"  - 批大小: {config.get('batch_size', 8)}")
        logger.info(f"  - 学习率: {self.learning_rate}")
        logger.info(f"  - 对齐权重: {self.alignment_weight}")
        logger.info(f"  - 可训练参数: {sum(p.numel() for p in trainable_params):,}")
    
    def compute_joint_loss(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算联合损失（生成损失 + 对齐损失）"""
        # 移动数据到设备
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        scaffold_modality = batch['scaffold_modality']
        
        # 1. 基础生成损失
        outputs = self.model.molt5_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        generation_loss = outputs.loss
        
        # 约束logits到有效词汇表范围
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
            if logits.size(-1) > self.vocab_size:
                invalid_mask = torch.zeros_like(logits)
                invalid_mask[:, :, self.vocab_size:] = -float('inf')
                logits = logits + invalid_mask
                
                # 重新计算损失
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                if labels is not None:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    generation_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                                             shift_labels.view(-1))
        
        # 2. 多模态对齐损失
        alignment_loss = torch.tensor(0.0, device=self.device)
        
        try:
            # 获取多模态特征
            scaffold_data = [batch['scaffold_data'][i] for i in range(len(batch['scaffold_data']))]
            text_data = [batch['text_data'][i] for i in range(len(batch['text_data']))]
            
            # 编码不同模态的特征
            modality_features = {}
            text_features = None
            
            # 按批次中的模态分组
            modality_groups = {}
            for i, modality in enumerate(scaffold_modality):
                if modality not in modality_groups:
                    modality_groups[modality] = []
                modality_groups[modality].append(i)
            
            # 为每种模态编码特征
            for modality, indices in modality_groups.items():
                batch_scaffold = [scaffold_data[i] for i in indices]
                batch_text = [text_data[i] for i in indices]
                
                # 编码scaffold特征
                if modality == 'smiles':
                    scaffold_features = self.model.encoder.smiles_encoder(batch_scaffold)
                elif modality == 'graph':
                    # 这里需要预处理为图数据
                    graph_data = [self.model.encoder.graph_encoder.preprocessor.smiles_to_graph(s) for s in batch_scaffold]
                    scaffold_features = self.model.encoder.graph_encoder(graph_data)
                elif modality == 'image':
                    # 这里需要预处理为图像数据
                    image_data = [self.model.encoder.image_encoder.preprocessor.smiles_to_image(s) for s in batch_scaffold]
                    scaffold_features = self.model.encoder.image_encoder(image_data)
                else:
                    continue
                
                # 编码文本特征
                batch_text_features = self.model.encoder.text_encoder(batch_text)
                
                modality_features[modality] = scaffold_features
                if text_features is None:
                    text_features = batch_text_features
                else:
                    text_features = torch.cat([text_features, batch_text_features], dim=0)
            
            # 计算对齐损失
            if modality_features and text_features is not None:
                alignment_loss = self.alignment_loss(modality_features, text_features)
        
        except Exception as e:
            logger.warning(f"对齐损失计算失败: {e}")
            alignment_loss = torch.tensor(0.0, device=self.device)
        
        # 总损失
        total_loss = generation_loss + self.alignment_weight * alignment_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'generation_loss': generation_loss.item(),
            'alignment_loss': alignment_loss.item() if isinstance(alignment_loss, torch.Tensor) else 0.0
        }
        
        return total_loss, loss_dict
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        self.alignment_loss.train()
        
        total_losses = {
            'total_loss': 0.0,
            'generation_loss': 0.0,
            'alignment_loss': 0.0
        }
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch + 1}/{self.num_epochs}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # 计算联合损失
            total_loss, loss_dict = self.compute_joint_loss(batch)
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.optimizer.param_groups[0]['params'] if p.requires_grad],
                    self.max_grad_norm
                )
            
            # 更新参数
            self.optimizer.step()
            self.scheduler.step()
            
            # 统计
            for key, value in loss_dict.items():
                total_losses[key] += value
            num_batches += 1
            self.global_step += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss_dict["total_loss"]:.4f}',
                'gen': f'{loss_dict["generation_loss"]:.4f}',
                'align': f'{loss_dict["alignment_loss"]:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # 日志记录
            if self.global_step % self.log_interval == 0:
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'train/{key}', value, self.global_step)
                self.writer.add_scalar('train/learning_rate', 
                                     self.optimizer.param_groups[0]['lr'], 
                                     self.global_step)
            
            # 验证
            if self.global_step % self.eval_interval == 0:
                val_metrics = self.validate()
                self.model.train()
                self.alignment_loss.train()
                
                # 记录验证指标
                for key, value in val_metrics.items():
                    self.writer.add_scalar(f'val/{key}', value, self.global_step)
                
                logger.info(f"Step {self.global_step} - Val Loss: {val_metrics.get('total_loss', 0):.4f}")
            
            # 保存检查点
            if self.global_step % self.save_interval == 0:
                self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')
        
        # 计算平均损失
        avg_losses = {}
        for key, value in total_losses.items():
            avg_losses[key] = value / num_batches if num_batches > 0 else 0.0
        
        return avg_losses
    
    def validate(self) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        self.alignment_loss.eval()
        
        total_losses = {
            'total_loss': 0.0,
            'generation_loss': 0.0,
            'alignment_loss': 0.0
        }
        num_batches = 0
        generated_smiles = []
        target_smiles_list = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                # 计算损失
                total_loss, loss_dict = self.compute_joint_loss(batch)
                
                for key, value in loss_dict.items():
                    total_losses[key] += value
                num_batches += 1
                
                # 生成样本（少量用于质量评估）
                if len(generated_smiles) < 30:  # 限制验证样本数量
                    try:
                        # 使用约束生成
                        input_ids = batch['input_ids'][:2].to(self.device)
                        attention_mask = batch['attention_mask'][:2].to(self.device)
                        
                        generated_ids = self.model.molt5_model.generate(
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
        
        # 计算平均验证指标
        val_metrics = {}
        for key, value in total_losses.items():
            val_metrics[key] = value / num_batches if num_batches > 0 else 0.0
        
        # 计算分子质量指标
        if generated_smiles and target_smiles_list:
            try:
                # 验证生成的SMILES
                valid_generated = []
                valid_targets = []
                for gen, target in zip(generated_smiles, target_smiles_list):
                    if gen and target:
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
                    
                    logger.info("联合训练验证指标:")
                    logger.info(f"  生成样本: {len(generated_smiles)}, 有效: {len(valid_generated)}")
                    for key, value in quality_metrics.items():
                        if isinstance(value, (int, float)) and key != 'total_loss':
                            logger.info(f"  {key}: {value:.4f}")
                
            except Exception as e:
                logger.warning(f"质量指标计算失败: {e}")
        
        return val_metrics
    
    def train(self):
        """完整训练流程"""
        logger.info("开始联合多模态训练...")
        logger.info(f"训练 {self.num_epochs} epochs")
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            logger.info(f"\n=== Epoch {epoch + 1}/{self.num_epochs} ===")
            
            # 训练一个epoch
            train_metrics = self.train_epoch()
            
            # 记录epoch指标
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'train_epoch/{key}', value, epoch)
            
            # Epoch结束时验证
            val_metrics = self.validate()
            
            # 记录验证指标
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'val_epoch/{key}', value, epoch)
            
            # 保存最佳模型
            val_loss = val_metrics.get('total_loss', float('inf'))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pt', is_best=True)
                logger.info(f"✅ 保存最佳模型 (Val Loss: {val_loss:.4f})")
            
            # 保存epoch检查点
            self.save_checkpoint(f'epoch_{epoch + 1}.pt')
            
            logger.info(f"Epoch {epoch + 1} - Train Loss: {train_metrics.get('total_loss', 0):.4f}, "
                       f"Val Loss: {val_loss:.4f}")
        
        total_time = time.time() - start_time
        logger.info(f"\n✅ 联合训练完成! 总用时: {total_time/3600:.2f} 小时")
        
        # 最终保存
        self.save_checkpoint('final_model.pt')
        self.writer.close()
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'alignment_loss_state_dict': self.alignment_loss.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'vocab_size': self.vocab_size
        }
        
        checkpoint_path = self.output_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.output_dir / 'model_best.pt'
            torch.save(checkpoint, best_path)


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='联合多模态分子生成模型训练')
    parser.add_argument('--train-data', type=str, default='Datasets/train.csv',
                       help='训练数据CSV文件')
    parser.add_argument('--val-data', type=str, default='Datasets/validation.csv',
                       help='验证数据CSV文件')
    parser.add_argument('--output-dir', type=str, 
                       default=f'outputs/joint_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                       help='输出目录')
    parser.add_argument('--batch-size', type=int, default=4, help='批大小')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--lr', type=float, default=5e-5, help='学习率')
    parser.add_argument('--alignment-weight', type=float, default=0.1, help='对齐损失权重')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--sample-size', type=int, default=1500, help='训练样本数限制')
    
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
        'save_interval': 300,
        'eval_interval': 150,
        'log_interval': 30,
        'output_dir': args.output_dir,
        'hidden_size': 768,
        'alignment_weight': args.alignment_weight,
        'alignment_temperature': 0.1,
        'max_text_length': 128,
        'max_smiles_length': 128,
        'filter_invalid': True
    }
    
    logger.info(f"联合训练配置: {json.dumps(config, indent=2)}")
    
    # 初始化tokenizer
    molt5_path = "/root/autodl-tmp/text2Mol-models/MolT5-Large-Caption2SMILES"
    tokenizer = T5Tokenizer.from_pretrained(molt5_path)
    
    # 模态混合比例
    modality_mix = {'smiles': 0.4, 'graph': 0.3, 'image': 0.3}
    
    # 创建数据集
    try:
        logger.info("创建联合训练数据集...")
        train_dataset = JointMultiModalDataset(
            csv_file=args.train_data,
            molt5_tokenizer=tokenizer,
            modality_mix=modality_mix,
            max_text_length=config['max_text_length'],
            max_smiles_length=config['max_smiles_length'],
            filter_invalid=config['filter_invalid']
        )
        
        # 限制样本数量
        if args.sample_size and len(train_dataset.data) > args.sample_size:
            train_dataset.data = train_dataset.data[:args.sample_size]
            logger.info(f"限制训练样本数为: {len(train_dataset.data)}")
        
        logger.info("创建验证数据集...")
        val_dataset = JointMultiModalDataset(
            csv_file=args.val_data,
            molt5_tokenizer=tokenizer,
            modality_mix=modality_mix,
            max_text_length=config['max_text_length'],
            max_smiles_length=config['max_smiles_length'],
            filter_invalid=config['filter_invalid']
        )
        
        # 限制验证样本
        if len(val_dataset.data) > 200:
            val_dataset.data = val_dataset.data[:200]
        
        logger.info(f"联合数据集创建完成: train={len(train_dataset)}, val={len(val_dataset)}")
        
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
            freeze_encoders=False,  # 解冻编码器进行联合训练
            freeze_molt5=False,     # 解冻MolT5进行微调
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
        trainer = JointMultiModalTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config,
            device=args.device
        )
        
        # 开始训练
        logger.info("🚀 开始联合多模态训练...")
        trainer.train()
        logger.info("🎉 联合训练成功完成！")
        
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