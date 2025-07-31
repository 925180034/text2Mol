"""
Training orchestration for scaffold-based molecular generation.

This module provides comprehensive training utilities with support for:
- Multi-modal training
- Scaffold preservation
- Progressive learning
- Distributed training
"""

import logging
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
import wandb

from ..models.core_model import ScaffoldBasedMolT5Generator
from ..data.collate import create_data_loader
from .loss_functions import ScaffoldLoss, MultiModalLoss
from .metrics import MolecularMetrics, GenerationMetrics
from ..utils.scaffold_utils import ScaffoldExtractor
from ..utils.mol_utils import MolecularUtils

logger = logging.getLogger(__name__)

class ScaffoldMolTrainer:
    """
    Comprehensive trainer for scaffold-based molecular generation.
    """
    
    def __init__(self,
                 model: ScaffoldBasedMolT5Generator,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 config: Dict[str, Any],
                 output_dir: str = 'outputs',
                 device: Optional[torch.device] = None,
                 logger_type: str = 'tensorboard'):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader  
            config: Training configuration
            output_dir: Output directory for checkpoints and logs
            device: Training device
            logger_type: Logging type ('tensorboard', 'wandb', 'none')
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device setup
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training parameters
        self.num_epochs = config.get('num_epochs', 100)
        self.learning_rate = config.get('learning_rate', 5e-5)
        self.weight_decay = config.get('weight_decay', 0.01)
        self.warmup_steps = config.get('warmup_steps', 1000)
        self.gradient_clip_norm = config.get('gradient_clip_norm', 1.0)
        self.accumulation_steps = config.get('gradient_accumulation_steps', 1)
        
        # Evaluation parameters
        self.eval_steps = config.get('eval_steps', 1000)
        self.save_steps = config.get('save_steps', 5000)
        self.logging_steps = config.get('logging_steps', 100)
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Loss functions and metrics
        self.loss_fn = self._create_loss_function()
        self.metrics = MolecularMetrics()
        self.generation_metrics = GenerationMetrics()
        
        # Logging setup
        self.logger_type = logger_type
        self._setup_logging()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.best_metrics = {}
        
        # Scaffold extractor for evaluation
        self.scaffold_extractor = ScaffoldExtractor()
        
        logger.info(f"Trainer initialized. Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with different learning rates for different components."""
        # Separate parameters for different learning rates
        molt5_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'molt5' in name:
                molt5_params.append(param)
            else:
                other_params.append(param)
        
        # Use lower learning rate for pre-trained MolT5
        param_groups = [
            {'params': molt5_params, 'lr': self.learning_rate * 0.1},
            {'params': other_params, 'lr': self.learning_rate}
        ]
        
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.weight_decay,
            eps=1e-8
        )
        
        return optimizer
    
    def _create_scheduler(self) -> Any:
        """Create learning rate scheduler."""
        total_steps = len(self.train_dataloader) * self.num_epochs // self.accumulation_steps
        
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        
        return scheduler
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function."""
        loss_config = self.config.get('loss', {})
        
        if loss_config.get('type') == 'scaffold':
            return ScaffoldLoss(loss_config)
        elif loss_config.get('type') == 'multimodal':
            return MultiModalLoss(loss_config)
        else:
            # Default: cross-entropy for SMILES generation
            return nn.CrossEntropyLoss(ignore_index=-100)
    
    def _setup_logging(self):
        """Setup logging infrastructure."""
        if self.logger_type == 'tensorboard':
            self.writer = SummaryWriter(log_dir=self.output_dir / 'logs')
        elif self.logger_type == 'wandb':
            wandb.init(
                project=self.config.get('project_name', 'scaffold-mol-generation'),
                config=self.config,
                dir=str(self.output_dir)
            )
            self.writer = None
        else:
            self.writer = None
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns:
            Training results and final metrics
        """
        logger.info("Starting training...")
        start_time = time.time()
        
        self.model.train()
        training_loss = 0.0
        
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            epoch_loss = self._train_epoch()
            
            # Validation
            if epoch % self.config.get('eval_epochs', 1) == 0:
                val_results = self._validate()
                
                # Save best model
                if val_results['loss'] < self.best_val_loss:
                    self.best_val_loss = val_results['loss']
                    self.best_metrics = val_results
                    self._save_checkpoint('best_model.pt')
                
                # Log results
                self._log_metrics({
                    'epoch': epoch,
                    'train_loss': epoch_loss,
                    **val_results
                })
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_epochs', 10) == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
        
        total_time = time.time() - start_time
        
        # Final evaluation
        final_results = self._final_evaluation()
        
        # Save final model
        self._save_checkpoint('final_model.pt')
        
        results = {
            'training_time': total_time,
            'best_validation_loss': self.best_val_loss,
            'best_metrics': self.best_metrics,
            'final_results': final_results
        }
        
        logger.info(f"Training completed in {total_time:.2f}s")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return results
    
    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for step, batch in enumerate(self.train_dataloader):
            # Move batch to device
            batch = self._move_to_device(batch)
            
            # Forward pass
            outputs = self.model(batch, mode='train')
            
            # Compute loss
            if isinstance(outputs, torch.Tensor):
                loss = outputs
            else:
                loss = self.loss_fn(outputs, batch)
            
            # Gradient accumulation
            loss = loss / self.accumulation_steps
            loss.backward()
            
            if (step + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.gradient_clip_norm
                )
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.logging_steps == 0:
                    self._log_training_step(loss.item() * self.accumulation_steps)
                
                # Validation during training
                if self.global_step % self.eval_steps == 0:
                    val_results = self._validate()
                    self._log_metrics(val_results)
                    self.model.train()  # Return to training mode
            
            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate(self) -> Dict[str, Any]:
        """Validation step."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = self._move_to_device(batch)
                
                # Forward pass
                outputs = self.model(batch, mode='train')
                
                # Compute loss
                if isinstance(outputs, torch.Tensor):
                    loss = outputs
                else:
                    loss = self.loss_fn(outputs, batch)
                
                total_loss += loss.item()
                
                # Collect predictions for metrics
                if 'raw_data' in batch:
                    # Generate predictions
                    predictions = self.model(batch, mode='inference')
                    if isinstance(predictions, list):
                        all_predictions.extend(predictions)
                        all_targets.extend(batch['raw_data']['smiles'])
        
        # Compute metrics
        val_loss = total_loss / len(self.val_dataloader)
        metrics = {}
        
        if all_predictions and all_targets:
            metrics = self.generation_metrics.compute_metrics(
                all_predictions, all_targets
            )
        
        results = {
            'loss': val_loss,
            **metrics
        }
        
        return results
    
    def _final_evaluation(self) -> Dict[str, Any]:
        """Comprehensive final evaluation."""
        logger.info("Running final evaluation...")
        
        self.model.eval()
        results = {}
        
        # Generate samples for evaluation
        test_samples = []
        generated_samples = []
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_dataloader):
                if i >= 10:  # Limit for final evaluation
                    break
                
                batch = self._move_to_device(batch)
                predictions = self.model(batch, mode='inference')
                
                if isinstance(predictions, list) and 'raw_data' in batch:
                    generated_samples.extend(predictions)
                    test_samples.extend(batch['raw_data']['smiles'])
        
        if generated_samples and test_samples:
            # Molecular metrics
            mol_metrics = self.generation_metrics.compute_comprehensive_metrics(
                generated_samples, test_samples
            )
            results.update(mol_metrics)
            
            # Scaffold preservation analysis
            scaffold_results = self._evaluate_scaffold_preservation(
                generated_samples, test_samples
            )
            results.update(scaffold_results)
        
        return results
    
    def _evaluate_scaffold_preservation(self, generated: List[str], 
                                      targets: List[str]) -> Dict[str, float]:
        """Evaluate scaffold preservation in generated molecules."""
        scaffold_preserved = 0
        valid_pairs = 0
        
        for gen, target in zip(generated, targets):
            if not MolecularUtils.validate_smiles(gen) or not MolecularUtils.validate_smiles(target):
                continue
                
            valid_pairs += 1
            
            # Extract scaffolds
            gen_scaffold = self.scaffold_extractor.get_murcko_scaffold(gen)
            target_scaffold = self.scaffold_extractor.get_murcko_scaffold(target)
            
            if gen_scaffold and target_scaffold and gen_scaffold == target_scaffold:
                scaffold_preserved += 1
        
        preservation_rate = scaffold_preserved / valid_pairs if valid_pairs > 0 else 0.0
        
        return {
            'scaffold_preservation_rate': preservation_rate,
            'valid_evaluation_pairs': valid_pairs
        }
    
    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to training device."""
        def move_item(item):
            if isinstance(item, torch.Tensor):
                return item.to(self.device)
            elif isinstance(item, dict):
                return {k: move_item(v) for k, v in item.items()}
            elif hasattr(item, 'to'):  # PyG Data objects
                return item.to(self.device)
            else:
                return item
        
        return {k: move_item(v) for k, v in batch.items()}
    
    def _log_training_step(self, loss: float):
        """Log training step metrics."""
        if self.writer:
            self.writer.add_scalar('train/loss', loss, self.global_step)
            self.writer.add_scalar('train/learning_rate', 
                                 self.scheduler.get_last_lr()[0], self.global_step)
        
        if self.logger_type == 'wandb':
            wandb.log({
                'train/loss': loss,
                'train/learning_rate': self.scheduler.get_last_lr()[0],
                'global_step': self.global_step
            })
    
    def _log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics to appropriate logger."""
        if self.writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'eval/{key}', value, self.global_step)
        
        if self.logger_type == 'wandb':
            wandb.log({f'eval/{k}': v for k, v in metrics.items() 
                      if isinstance(v, (int, float))})
        
        # Console logging
        logger.info(f"Step {self.global_step}: {metrics}")
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        checkpoint_path = self.output_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")


class MultiTaskTrainer(ScaffoldMolTrainer):
    """
    Multi-task trainer for simultaneous generation of multiple modalities.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Multi-task specific configuration
        self.task_weights = self.config.get('task_weights', {
            'smiles': 1.0,
            'graph': 0.5,
            'image': 0.3
        })
        
        self.active_tasks = self.config.get('active_tasks', ['smiles'])
        
        logger.info(f"Multi-task training with tasks: {self.active_tasks}")
        logger.info(f"Task weights: {self.task_weights}")
    
    def _train_epoch(self) -> float:
        """Multi-task training epoch."""
        self.model.train()
        task_losses = {task: 0.0 for task in self.active_tasks}
        total_loss = 0.0
        num_batches = 0
        
        for step, batch in enumerate(self.train_dataloader):
            batch = self._move_to_device(batch)
            total_batch_loss = 0.0
            
            # Train on each active task
            for task in self.active_tasks:
                outputs = self.model(batch, output_modality=task, mode='train')
                
                if isinstance(outputs, torch.Tensor):
                    task_loss = outputs
                else:
                    task_loss = self.loss_fn(outputs, batch)
                
                # Apply task weighting
                weighted_loss = task_loss * self.task_weights.get(task, 1.0)
                total_batch_loss += weighted_loss
                task_losses[task] += task_loss.item()
            
            # Gradient accumulation
            loss = total_batch_loss / self.accumulation_steps
            loss.backward()
            
            if (step + 1) % self.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.gradient_clip_norm
                )
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.logging_steps == 0:
                    self._log_multitask_step(task_losses, num_batches + 1)
            
            total_loss += total_batch_loss.item()
            num_batches += 1
        
        # Average losses
        avg_task_losses = {task: loss / num_batches 
                          for task, loss in task_losses.items()}
        
        logger.info(f"Epoch {self.epoch} task losses: {avg_task_losses}")
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _log_multitask_step(self, task_losses: Dict[str, float], num_batches: int):
        """Log multi-task training metrics."""
        avg_losses = {task: loss / num_batches for task, loss in task_losses.items()}
        
        if self.writer:
            for task, loss in avg_losses.items():
                self.writer.add_scalar(f'train/{task}_loss', loss, self.global_step)
        
        if self.logger_type == 'wandb':
            wandb.log({f'train/{task}_loss': loss for task, loss in avg_losses.items()})


def create_trainer(model: ScaffoldBasedMolT5Generator,
                  train_dataset: Any,
                  val_dataset: Any,
                  config: Dict[str, Any],
                  **kwargs) -> ScaffoldMolTrainer:
    """
    Factory function to create appropriate trainer.
    
    Args:
        model: Model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Training configuration
        **kwargs: Additional arguments
        
    Returns:
        Configured trainer instance
    """
    # Create data loaders
    train_loader = create_data_loader(
        train_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=True,
        num_workers=config.get('num_workers', 0),
        collate_type=config.get('collate_type', 'standard')
    )
    
    val_loader = create_data_loader(
        val_dataset,
        batch_size=config.get('eval_batch_size', 32),
        shuffle=False,
        num_workers=config.get('num_workers', 0),
        collate_type=config.get('collate_type', 'standard')
    )
    
    # Choose trainer type
    if config.get('multi_task', False):
        return MultiTaskTrainer(
            model, train_loader, val_loader, config, **kwargs
        )
    else:
        return ScaffoldMolTrainer(
            model, train_loader, val_loader, config, **kwargs
        )