#!/usr/bin/env python3
"""
修复版9种模态组合训练脚本
支持 (SMILES/Graph/Image) × (SMILES/Graph/Image) = 9种组合
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
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import pandas as pd

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from scaffold_mol_gen.models.end2end_model import End2EndMolecularGenerator
from scaffold_mol_gen.models.graph_decoder import MolecularGraphDecoder as GraphDecoder
from scaffold_mol_gen.models.image_decoder import MolecularImageDecoder as ImageDecoder
from scaffold_mol_gen.data.multimodal_dataset import MultiModalMolecularDataset
from scaffold_mol_gen.training.metrics import GenerationMetrics
from scaffold_mol_gen.utils.mol_utils import MolecularUtils
from scaffold_mol_gen.data.multimodal_preprocessor import MultiModalPreprocessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NineModalityTrainer:
    """9种模态组合训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info("初始化9模态训练器...")
        
        # 创建模型
        logger.info("创建端到端模型...")
        self.model = End2EndMolecularGenerator(
            hidden_size=768,
            molt5_path="/root/autodl-tmp/text2Mol-models/molt5-base",
            use_scibert=False,
            freeze_encoders=True,
            freeze_molt5=True,
            fusion_type='both',
            device=self.device
        )
        
        # 创建Graph和Image解码器
        logger.info("创建Graph和Image解码器...")
        self.graph_decoder = GraphDecoder(input_dim=768).to(self.device)
        self.image_decoder = ImageDecoder(input_dim=768).to(self.device)
        
        # 损失权重
        self.loss_weights = {
            'smiles': args.smiles_weight,
            'graph': args.graph_weight,
            'image': args.image_weight
        }
        
        # 创建数据加载器
        logger.info("创建数据加载器...")
        self.train_loader, self.val_loader = self.create_data_loaders()
        
        # 优化器
        all_params = (
            list(self.model.parameters()) + 
            list(self.graph_decoder.parameters()) + 
            list(self.image_decoder.parameters())
        )
        trainable_params = [p for p in all_params if p.requires_grad]
        
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=1e-5
        )
        
        # 混合精度训练
        self.scaler = GradScaler() if args.mixed_precision else None
        
        # 输出目录
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(self.output_dir / 'tensorboard')
        
        # 保存配置
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(vars(args), f, indent=2)
        
        logger.info(f"训练器初始化完成")
        logger.info(f"  设备: {self.device}")
        logger.info(f"  可训练参数: {sum(p.numel() for p in trainable_params):,}")
        logger.info(f"  训练批次: {len(self.train_loader)}")
    
    def create_data_loaders(self):
        """创建数据加载器"""
        # 训练集
        train_dataset = MultiModalMolecularDataset(
            csv_path=self.args.train_data,
            scaffold_modality='smiles',  # 默认SMILES，会在训练中转换
            filter_invalid=True
        )
        
        # 限制样本数
        if self.args.sample_size > 0:
            train_dataset.data = train_dataset.data[:self.args.sample_size]
        
        # 验证集
        val_dataset = MultiModalMolecularDataset(
            csv_path=self.args.val_data,
            scaffold_modality='smiles',
            filter_invalid=True
        )
        
        if self.args.sample_size > 0:
            val_dataset.data = val_dataset.data[:min(1000, self.args.sample_size//5)]
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        return train_loader, val_loader
    
    def prepare_multimodal_batch(self, batch):
        """准备多模态批次数据"""
        # 获取原始数据 - 修正键名
        scaffold_smiles = batch.get('scaffold_data')  # 正确的键名
        text = batch.get('text_data')  # 正确的键名
        target_smiles = batch.get('target_smiles')  # 这个键名是对的
        
        # 创建多模态数据
        preprocessor = MultiModalPreprocessor()
        
        batch_data = {
            'text': text,
            'target_smiles': target_smiles
        }
        
        # 准备三种scaffold模态
        if scaffold_smiles is not None:
            # SMILES模态
            batch_data['scaffold_smiles'] = scaffold_smiles
            
            # Graph模态 - 从SMILES转换
            try:
                graphs = []
                for smiles in scaffold_smiles:
                    graph = preprocessor.smiles_to_graph(smiles)
                    if graph is not None:
                        graphs.append(graph)
                if graphs:
                    from torch_geometric.data import Batch
                    batch_data['scaffold_graph'] = Batch.from_data_list(graphs)
            except Exception as e:
                logger.debug(f"Graph转换失败: {e}")
            
            # Image模态 - 从SMILES转换
            try:
                images = []
                for smiles in scaffold_smiles:
                    img = preprocessor.smiles_to_image(smiles)
                    if img is not None:
                        images.append(img)
                if images:
                    batch_data['scaffold_image'] = torch.stack([
                        torch.from_numpy(img).float() for img in images
                    ]).to(self.device)
            except Exception as e:
                logger.debug(f"Image转换失败: {e}")
        
        # 准备目标模态（用于Graph和Image解码器）
        if target_smiles is not None:
            # Target Graph
            try:
                target_graphs = []
                for smiles in target_smiles:
                    graph = preprocessor.smiles_to_graph(smiles)
                    if graph is not None:
                        target_graphs.append(graph)
                if target_graphs:
                    from torch_geometric.data import Batch
                    batch_data['target_graph'] = Batch.from_data_list(target_graphs)
            except:
                pass
            
            # Target Image
            try:
                target_images = []
                for smiles in target_smiles:
                    img = preprocessor.smiles_to_image(smiles)
                    if img is not None:
                        target_images.append(img)
                if target_images:
                    batch_data['target_image'] = torch.stack([
                        torch.from_numpy(img).float() for img in target_images
                    ]).to(self.device)
            except:
                pass
        
        return batch_data
    
    def compute_loss_for_combination(self, scaffold_data, text_data, target_data, 
                                    scaffold_modality, output_modality):
        """计算单个模态组合的损失"""
        try:
            # 获取融合特征
            output = self.model.forward(
                scaffold_data=scaffold_data,
                text_data=text_data,
                scaffold_modality=scaffold_modality,
                output_modality='smiles',  # 先生成SMILES
                target_smiles=target_data.get('target_smiles') if output_modality == 'smiles' else None
            )
            
            # 调试信息
            if not output:
                logger.debug(f"{scaffold_modality}->{output_modality}: output is None")
                return None
                
            if output_modality == 'smiles':
                # SMILES损失直接从model获得
                if 'loss' in output and output['loss'] is not None:
                    loss_value = output['loss']
                    # 确保loss是tensor并且有效
                    if torch.is_tensor(loss_value) and not torch.isnan(loss_value) and not torch.isinf(loss_value):
                        return loss_value
                    else:
                        logger.debug(f"{scaffold_modality}->smiles: 无效损失 {loss_value}")
                else:
                    logger.debug(f"{scaffold_modality}->smiles: 无loss字段或loss为None")
            
            elif output_modality == 'graph' and 'fused_features' in output:
                # Graph损失
                if 'target_graph' in target_data:
                    graph_loss = self.graph_decoder.compute_loss(
                        output['fused_features'],
                        target_data['target_graph']
                    )
                    if graph_loss is not None and torch.is_tensor(graph_loss):
                        return graph_loss
                    else:
                        logger.debug(f"{scaffold_modality}->graph: Graph损失计算失败")
                else:
                    logger.debug(f"{scaffold_modality}->graph: 无target_graph")
            
            elif output_modality == 'image' and 'fused_features' in output:
                # Image损失
                if 'target_image' in target_data:
                    image_loss = self.image_decoder.compute_loss(
                        output['fused_features'],
                        target_data['target_image']
                    )
                    if image_loss is not None and torch.is_tensor(image_loss):
                        return image_loss
                    else:
                        logger.debug(f"{scaffold_modality}->image: Image损失计算失败")
                else:
                    logger.debug(f"{scaffold_modality}->image: 无target_image")
        
        except Exception as e:
            logger.debug(f"损失计算异常 {scaffold_modality}->{output_modality}: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
        
        return None
    
    def compute_multi_task_loss(self, batch):
        """计算9种模态组合的多任务损失"""
        losses = {}
        total_loss = None
        
        # 9种组合
        combinations = [
            ('smiles', 'smiles'), ('smiles', 'graph'), ('smiles', 'image'),
            ('graph', 'smiles'), ('graph', 'graph'), ('graph', 'image'),
            ('image', 'smiles'), ('image', 'graph'), ('image', 'image')
        ]
        
        for scaffold_mod, output_mod in combinations:
            scaffold_key = f'scaffold_{scaffold_mod}'
            
            if scaffold_key not in batch:
                continue
            
            scaffold_data = batch[scaffold_key]
            if scaffold_data is None:
                continue
            
            # 计算损失
            loss = self.compute_loss_for_combination(
                scaffold_data=scaffold_data,
                text_data=batch['text'],
                target_data=batch,
                scaffold_modality=scaffold_mod,
                output_modality=output_mod
            )
            
            if loss is not None and torch.is_tensor(loss):
                # 应用权重
                weight = self.loss_weights.get(output_mod, 1.0)
                weighted_loss = loss * weight
                
                losses[f'{scaffold_mod}_to_{output_mod}'] = loss.item()
                
                # 累加到总损失
                if total_loss is None:
                    total_loss = weighted_loss
                else:
                    total_loss = total_loss + weighted_loss
        
        # 确保返回有效的tensor
        if total_loss is None:
            # 如果所有损失都失败，返回一个小的常数损失
            total_loss = torch.tensor(0.01, device=self.device, requires_grad=True)
            logger.warning("所有模态组合损失计算失败，使用默认损失")
        
        losses['total'] = total_loss
        return losses
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        self.graph_decoder.train()
        self.image_decoder.train()
        
        total_loss = 0
        loss_counts = {}
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.args.epochs}')
        
        for batch_idx, raw_batch in enumerate(progress_bar):
            # 准备多模态批次
            batch = self.prepare_multimodal_batch(raw_batch)
            
            # 计算损失
            if self.scaler:
                with autocast():
                    losses = self.compute_multi_task_loss(batch)
                    loss = losses['total']
                
                # 检查loss是否有效
                if torch.is_tensor(loss) and loss.requires_grad:
                    # 反向传播
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    
                    # 梯度累积
                    if (batch_idx + 1) % self.args.gradient_accumulation == 0:
                        # 只有在有梯度时才更新
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad] +
                            list(self.graph_decoder.parameters()) +
                            list(self.image_decoder.parameters()),
                            self.args.max_grad_norm
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    
                    total_loss += loss.item()
                else:
                    # 如果loss无效，跳过这个batch
                    logger.warning(f"Batch {batch_idx}: 跳过无效损失")
                    continue
                    
                    # 记录各模态损失
                    for key, value in losses.items():
                        if key != 'total' and isinstance(value, (int, float)):
                            if key not in loss_counts:
                                loss_counts[key] = []
                            loss_counts[key].append(value)
            else:
                losses = self.compute_multi_task_loss(batch)
                loss = losses['total']
                
                if torch.is_tensor(loss) and loss.requires_grad:
                    loss.backward()
                    
                    if (batch_idx + 1) % self.args.gradient_accumulation == 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad] +
                            list(self.graph_decoder.parameters()) +
                            list(self.image_decoder.parameters()),
                            self.args.max_grad_norm
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    
                    total_loss += loss.item()
            
            # 更新进度条
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # 定期记录
            if batch_idx % 100 == 0 and batch_idx > 0:
                logger.info(f"Batch {batch_idx}, Avg Loss: {avg_loss:.4f}")
                
                # 记录各模态平均损失
                for key, values in loss_counts.items():
                    if values:
                        avg = sum(values) / len(values)
                        logger.info(f"  {key}: {avg:.4f}")
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """验证"""
        self.model.eval()
        self.graph_decoder.eval()
        self.image_decoder.eval()
        
        total_loss = 0
        
        with torch.no_grad():
            for raw_batch in tqdm(self.val_loader, desc='Validation'):
                batch = self.prepare_multimodal_batch(raw_batch)
                losses = self.compute_multi_task_loss(batch)
                
                if 'total' in losses and torch.is_tensor(losses['total']):
                    total_loss += losses['total'].item()
        
        return total_loss / len(self.val_loader)
    
    def train(self):
        """主训练循环"""
        logger.info("=" * 60)
        logger.info("🚀 开始9种模态组合训练")
        logger.info(f"设备: {self.device}")
        logger.info(f"批大小: {self.args.batch_size}")
        logger.info(f"梯度累积: {self.args.gradient_accumulation}")
        logger.info(f"有效批大小: {self.args.batch_size * self.args.gradient_accumulation}")
        logger.info(f"混合精度: {self.args.mixed_precision}")
        logger.info("=" * 60)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.args.epochs):
            start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss = self.validate()
            
            epoch_time = time.time() - start_time
            
            logger.info(f"Epoch {epoch+1}/{self.args.epochs}: "
                       f"Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, "
                       f"Time: {epoch_time:.1f}s")
            
            # TensorBoard记录
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, is_best=True)
                logger.info(f"✅ 保存最佳模型，验证损失: {val_loss:.4f}")
            
            # 定期保存
            if (epoch + 1) % self.args.save_interval == 0:
                self.save_checkpoint(epoch, val_loss, is_best=False)
        
        logger.info("🎉 训练完成!")
        return {'best_val_loss': best_val_loss}
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'graph_decoder_state_dict': self.graph_decoder.state_dict(),
            'image_decoder_state_dict': self.image_decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        if is_best:
            path = self.output_dir / 'best_model.pth'
        else:
            path = self.output_dir / f'checkpoint_epoch_{epoch+1}.pth'
        
        torch.save(checkpoint, path)
        logger.info(f"保存检查点: {path}")


def main():
    parser = argparse.ArgumentParser(description='9种模态组合训练')
    
    # 数据参数
    parser.add_argument('--train-data', default='Datasets/train.csv',
                        help='训练数据路径')
    parser.add_argument('--val-data', default='Datasets/validation.csv',
                        help='验证数据路径')
    parser.add_argument('--sample-size', type=int, default=0,
                        help='样本数量限制（0=全部）')
    
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=8,
                        help='批大小')
    parser.add_argument('--gradient-accumulation', type=int, default=1,
                        help='梯度累积步数')
    parser.add_argument('--epochs', type=int, default=5,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='学习率')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                        help='梯度裁剪')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='启用混合精度训练')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='数据加载进程数')
    
    # 损失权重
    parser.add_argument('--smiles-weight', type=float, default=1.0,
                        help='SMILES损失权重')
    parser.add_argument('--graph-weight', type=float, default=0.7,
                        help='Graph损失权重')
    parser.add_argument('--image-weight', type=float, default=0.5,
                        help='Image损失权重')
    
    # 输出参数
    parser.add_argument('--output-dir', 
                        default='/root/autodl-tmp/text2Mol-outputs/nine_modal_training',
                        help='输出目录')
    parser.add_argument('--save-interval', type=int, default=1,
                        help='保存间隔')
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = NineModalityTrainer(args)
    
    # 开始训练
    stats = trainer.train()
    
    logger.info(f"最终结果: {stats}")


if __name__ == "__main__":
    main()