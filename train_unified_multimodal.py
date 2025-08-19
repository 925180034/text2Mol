#!/usr/bin/env python3
"""
统一多模态训练脚本
支持9种输入输出组合的完整训练
基于当前的架构组件，实现真正的多模态输出训练
"""

import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb
from pathlib import Path
from typing import Dict, List, Any, Optional

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from scaffold_mol_gen.models.end2end_model import End2EndMolecularGenerator
from scaffold_mol_gen.models.graph_decoder import MolecularGraphDecoder, GraphDecoderLoss
from scaffold_mol_gen.models.image_decoder import MolecularImageDecoder, ImageDecoderLoss
from scaffold_mol_gen.models.output_decoders import OutputDecoder
from scaffold_mol_gen.data.dataset import MultiModalMolecularDataset
from scaffold_mol_gen.training.metrics import MolecularGenerationMetrics

logger = logging.getLogger(__name__)

class UnifiedMultiModalTrainer:
    """统一的多模态训练器"""
    
    def __init__(self, 
                 model: End2EndMolecularGenerator,
                 graph_decoder: MolecularGraphDecoder,
                 image_decoder: MolecularImageDecoder,
                 output_decoder: OutputDecoder,
                 device: str = 'cuda'):
        """
        Args:
            model: 端到端模型（用于SMILES生成）
            graph_decoder: 图解码器（768维→图）
            image_decoder: 图像解码器（768维→图像）
            output_decoder: 输出转换器（SMILES→其他模态）
            device: 训练设备
        """
        self.model = model.to(device)
        self.graph_decoder = graph_decoder.to(device)
        self.image_decoder = image_decoder.to(device)
        self.output_decoder = output_decoder
        self.device = device
        
        # 损失函数
        self.graph_loss_fn = GraphDecoderLoss()
        self.image_loss_fn = ImageDecoderLoss()
        self.metrics = MolecularGenerationMetrics()
        
        # 任务权重
        self.task_weights = {
            'smiles': 1.0,
            'graph': 0.5, 
            'image': 0.3
        }
        
        logger.info("初始化统一多模态训练器")
    
    def forward_training_step(self, 
                            batch: Dict[str, Any],
                            current_epoch: int,
                            total_epochs: int) -> Dict[str, torch.Tensor]:
        """
        训练前向步骤
        
        Args:
            batch: 批次数据
            current_epoch: 当前轮次
            total_epochs: 总轮次
            
        Returns:
            损失字典
        """
        losses = {}
        
        # 1. 获取批次数据
        scaffold_data = batch['scaffold_smiles']
        text_data = batch['text']
        target_smiles = batch['target_smiles']
        scaffold_modality = batch.get('scaffold_modality', 'smiles')
        
        # 2. 端到端模型前向传播（获取融合特征）
        output_dict = self.model(
            scaffold_data=scaffold_data,
            text_data=text_data,
            scaffold_modality=scaffold_modality,
            target_smiles=target_smiles,
            output_modality='smiles'
        )
        
        # 3. SMILES生成损失
        if 'loss' in output_dict:
            losses['smiles_generation'] = output_dict['loss']
        
        # 4. 获取融合特征用于其他模态解码
        fused_features = output_dict.get('fused_features')
        
        if fused_features is not None:
            # 5. 图解码损失
            if 'target_graph' in batch:
                graph_predictions = self.graph_decoder(fused_features)
                graph_loss, graph_loss_dict = self.graph_loss_fn(
                    graph_predictions, 
                    batch['target_graph']
                )
                losses['graph_decode'] = graph_loss
                
                # 记录图解码细节
                for k, v in graph_loss_dict.items():
                    losses[f'graph_{k}'] = torch.tensor(v, device=self.device)
            
            # 6. 图像解码损失
            if 'target_image' in batch:
                generated_images = self.image_decoder(fused_features)
                image_loss, image_loss_dict = self.image_loss_fn(
                    generated_images,
                    batch['target_image']
                )
                losses['image_decode'] = image_loss
                
                # 记录图像解码细节
                for k, v in image_loss_dict.items():
                    losses[f'image_{k}'] = torch.tensor(v, device=self.device)
        
        # 7. 计算加权总损失
        total_loss = torch.tensor(0.0, device=self.device)
        
        # 基础SMILES生成损失
        if 'smiles_generation' in losses:
            total_loss += self.task_weights['smiles'] * losses['smiles_generation']
        
        # 图解码损失
        if 'graph_decode' in losses:
            # 逐渐增加图解码权重
            graph_weight = self.task_weights['graph'] * min(1.0, current_epoch / (total_epochs * 0.3))
            total_loss += graph_weight * losses['graph_decode']
        
        # 图像解码损失
        if 'image_decode' in losses:
            # 后期加入图像解码
            image_weight = self.task_weights['image'] * max(0.0, (current_epoch - total_epochs * 0.5) / (total_epochs * 0.5))
            total_loss += image_weight * losses['image_decode']
        
        losses['total'] = total_loss
        
        return losses
    
    def train_epoch(self, 
                   dataloader: DataLoader,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   total_epochs: int) -> Dict[str, float]:
        """训练一个epoch"""
        
        self.model.train()
        self.graph_decoder.train()
        self.image_decoder.train()
        
        epoch_losses = {}
        num_batches = len(dataloader)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 移动数据到设备
            batch = self._move_batch_to_device(batch)
            
            # 前向传播
            losses = self.forward_training_step(batch, epoch, total_epochs)
            
            # 反向传播
            optimizer.zero_grad()
            losses['total'].backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + 
                list(self.graph_decoder.parameters()) + 
                list(self.image_decoder.parameters()), 
                max_norm=1.0
            )
            
            optimizer.step()
            
            # 记录损失
            for key, value in losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = []
                epoch_losses[key].append(value.item() if torch.is_tensor(value) else value)
            
            # 更新进度条
            if batch_idx % 10 == 0:
                current_loss = losses['total'].item()
                progress_bar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'smiles': f"{losses.get('smiles_generation', 0):.3f}",
                    'graph': f"{losses.get('graph_decode', 0):.3f}",
                    'image': f"{losses.get('image_decode', 0):.3f}"
                })
        
        # 计算平均损失
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        
        return avg_losses
    
    def validate_epoch(self, 
                      dataloader: DataLoader,
                      epoch: int) -> Dict[str, float]:
        """验证一个epoch"""
        
        self.model.eval()
        self.graph_decoder.eval()
        self.image_decoder.eval()
        
        val_losses = {}
        all_generated_smiles = []
        all_target_smiles = []
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc=f"Validation Epoch {epoch+1}")
            
            for batch in progress_bar:
                batch = self._move_batch_to_device(batch)
                
                # 生成SMILES用于评估
                try:
                    generated_smiles = self.model.generate(
                        scaffold_data=batch['scaffold_smiles'],
                        text_data=batch['text'],
                        scaffold_modality=batch.get('scaffold_modality', 'smiles'),
                        output_modality='smiles',
                        num_beams=3,
                        temperature=0.8,
                        max_length=128,
                        num_return_sequences=1
                    )
                    
                    all_generated_smiles.extend(generated_smiles)
                    all_target_smiles.extend(batch['target_smiles'])
                    
                except Exception as e:
                    logger.warning(f"生成失败: {e}")
                    # 添加无效占位符
                    all_generated_smiles.extend(['[INVALID]'] * len(batch['target_smiles']))
                    all_target_smiles.extend(batch['target_smiles'])
        
        # 计算分子生成指标
        if all_generated_smiles and all_target_smiles:
            metrics = self.metrics.compute_all_metrics(
                generated=all_generated_smiles,
                targets=all_target_smiles,
                descriptions=None  # 如果有描述可以加上
            )
            
            val_losses.update(metrics)
            
            logger.info(f"Validation Metrics:")
            logger.info(f"  Validity: {metrics.get('validity', 0):.3f}")
            logger.info(f"  Uniqueness: {metrics.get('uniqueness', 0):.3f}")
            logger.info(f"  BLEU: {metrics.get('bleu_score', 0):.3f}")
            logger.info(f"  Exact Match: {metrics.get('exact_match', 0):.3f}")
        
        return val_losses
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """移动批次数据到设备"""
        device_batch = {}
        
        for key, value in batch.items():
            if torch.is_tensor(value):
                device_batch[key] = value.to(self.device)
            elif isinstance(value, list) and len(value) > 0 and torch.is_tensor(value[0]):
                device_batch[key] = [v.to(self.device) for v in value]
            else:
                device_batch[key] = value
        
        return device_batch


def create_models(args) -> tuple:
    """创建所有模型组件"""
    
    # 1. 端到端模型（用于SMILES生成）
    logger.info("创建端到端模型...")
    e2e_model = End2EndMolecularGenerator(
        hidden_size=args.hidden_size,
        molt5_path=args.molt5_path,
        use_scibert=args.use_scibert,
        freeze_encoders=not args.unfreeze_encoders,
        freeze_molt5=not args.unfreeze_molt5,
        fusion_type=args.fusion_type,
        device=args.device
    )
    
    # 2. 图解码器
    logger.info("创建图解码器...")
    graph_decoder = MolecularGraphDecoder(
        input_dim=args.hidden_size,
        max_atoms=args.max_atoms,
        hidden_dim=args.graph_hidden_dim,
        num_layers=3,
        dropout=args.dropout
    )
    
    # 3. 图像解码器
    logger.info("创建图像解码器...")
    image_decoder = MolecularImageDecoder(
        input_dim=args.hidden_size,
        image_size=args.image_size,
        channels=3,
        hidden_dim=args.image_hidden_dim,
        num_layers=4,
        dropout=args.dropout
    )
    
    # 4. 输出转换器
    output_decoder = OutputDecoder()
    
    return e2e_model, graph_decoder, image_decoder, output_decoder


def create_dataloader(args, split: str) -> DataLoader:
    """创建数据加载器"""
    
    # 数据文件路径
    data_files = {
        'train': args.train_data,
        'validation': args.val_data,
        'test': args.test_data
    }
    
    if split not in data_files or not data_files[split]:
        raise ValueError(f"数据文件未指定: {split}")
    
    # 创建数据集
    dataset = MultiModalMolecularDataset(
        data_path=data_files[split],
        max_length=args.max_length,
        include_graph_targets=True,  # 包含图目标
        include_image_targets=True,   # 包含图像目标
        cache_dir=args.cache_dir
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(split == 'train'),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader


def main():
    parser = argparse.ArgumentParser(description="统一多模态分子生成训练")
    
    # 模型参数
    parser.add_argument('--hidden-size', type=int, default=768, help='隐藏层维度')
    parser.add_argument('--graph-hidden-dim', type=int, default=512, help='图解码器隐藏维度')
    parser.add_argument('--image-hidden-dim', type=int, default=512, help='图像解码器隐藏维度')
    parser.add_argument('--max-atoms', type=int, default=100, help='最大原子数')
    parser.add_argument('--image-size', type=int, default=224, help='图像尺寸')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    parser.add_argument('--fusion-type', type=str, default='both', choices=['attention', 'gated', 'both'])
    
    # 数据参数
    parser.add_argument('--train-data', type=str, default='Datasets/train.csv', help='训练数据')
    parser.add_argument('--val-data', type=str, default='Datasets/validation.csv', help='验证数据')
    parser.add_argument('--test-data', type=str, default='Datasets/test.csv', help='测试数据')
    parser.add_argument('--max-length', type=int, default=128, help='最大序列长度')
    parser.add_argument('--cache-dir', type=str, default='/root/autodl-tmp/text2Mol-cache', help='缓存目录')
    
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=8, help='批次大小')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮次')
    parser.add_argument('--lr', type=float, default=5e-5, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载器工作进程')
    
    # 模型路径
    parser.add_argument('--molt5-path', type=str, 
                       default='/root/autodl-tmp/text2Mol-models/molt5-base', 
                       help='MolT5模型路径')
    parser.add_argument('--output-dir', type=str, 
                       default='/root/autodl-tmp/text2Mol-outputs/unified_training',
                       help='输出目录')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    parser.add_argument('--use-scibert', action='store_true', help='使用SciBERT')
    parser.add_argument('--unfreeze-encoders', action='store_true', help='解冻编码器')
    parser.add_argument('--unfreeze-molt5', action='store_true', help='解冻MolT5')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点')
    parser.add_argument('--wandb-project', type=str, default='text2mol-unified', help='WandB项目名')
    parser.add_argument('--sample-size', type=int, help='限制训练样本数量（调试用）')
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    if torch.cuda.is_available():
        args.device = 'cuda'
        logger.info(f"使用GPU: {torch.cuda.get_device_name()}")
    else:
        args.device = 'cpu'
        logger.info("使用CPU训练")
    
    # 初始化WandB
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"unified-training-{args.fusion_type}"
        )
    
    try:
        # 创建模型
        e2e_model, graph_decoder, image_decoder, output_decoder = create_models(args)
        
        # 创建训练器
        trainer = UnifiedMultiModalTrainer(
            model=e2e_model,
            graph_decoder=graph_decoder,
            image_decoder=image_decoder,
            output_decoder=output_decoder,
            device=args.device
        )
        
        # 创建数据加载器
        logger.info("创建数据加载器...")
        train_loader = create_dataloader(args, 'train')
        val_loader = create_dataloader(args, 'validation')
        
        logger.info(f"训练样本: {len(train_loader.dataset)}")
        logger.info(f"验证样本: {len(val_loader.dataset)}")
        
        # 创建优化器和调度器
        all_params = (
            list(e2e_model.parameters()) + 
            list(graph_decoder.parameters()) + 
            list(image_decoder.parameters())
        )
        
        optimizer = AdamW(
            [p for p in all_params if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01
        )
        
        # 训练循环
        best_validity = 0.0
        
        for epoch in range(args.epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch+1}/{args.epochs}")
            logger.info(f"{'='*50}")
            
            # 训练
            train_losses = trainer.train_epoch(train_loader, optimizer, epoch, args.epochs)
            
            # 验证
            val_metrics = trainer.validate_epoch(val_loader, epoch)
            
            # 更新学习率
            scheduler.step()
            
            # 记录日志
            logger.info(f"训练损失: {train_losses.get('total', 0):.4f}")
            logger.info(f"验证指标: validity={val_metrics.get('validity', 0):.3f}")
            
            # WandB记录
            if args.wandb_project:
                log_dict = {}
                # 训练损失
                for k, v in train_losses.items():
                    log_dict[f'train/{k}'] = v
                # 验证指标
                for k, v in val_metrics.items():
                    log_dict[f'val/{k}'] = v
                log_dict['epoch'] = epoch
                log_dict['lr'] = scheduler.get_last_lr()[0]
                
                wandb.log(log_dict)
            
            # 保存最佳模型
            current_validity = val_metrics.get('validity', 0)
            if current_validity > best_validity:
                best_validity = current_validity
                
                checkpoint = {
                    'epoch': epoch,
                    'e2e_model_state_dict': e2e_model.state_dict(),
                    'graph_decoder_state_dict': graph_decoder.state_dict(),
                    'image_decoder_state_dict': image_decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_validity': best_validity,
                    'args': vars(args)
                }
                
                torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pth'))
                logger.info(f"保存最佳模型 (validity: {best_validity:.3f})")
            
            # 定期保存检查点
            if (epoch + 1) % 5 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'e2e_model_state_dict': e2e_model.state_dict(),
                    'graph_decoder_state_dict': graph_decoder.state_dict(),
                    'image_decoder_state_dict': image_decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_validity': best_validity,
                    'args': vars(args)
                }
                
                torch.save(checkpoint, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        logger.info(f"\n训练完成！最佳Validity: {best_validity:.3f}")
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        raise
    
    finally:
        if args.wandb_project:
            wandb.finish()


if __name__ == "__main__":
    main()