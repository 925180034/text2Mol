#!/usr/bin/env python3
"""
Fast Training Script - Optimized for 2-4 hours training time
Uses multiple optimization techniques to achieve 10-20x speedup
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from scaffold_mol_gen.models.end2end_model import End2EndMolecularGenerator
from scaffold_mol_gen.data.dataset import ScaffoldDataset
from scaffold_mol_gen.training.metrics import MolecularMetrics

# Setup minimal logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class FastTrainer:
    """Optimized trainer for rapid training"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Using device: {self.device}")
        
        # Enable optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Initialize model with optimizations
        self.model = self._create_optimized_model()
        
        # Compile model for speed (PyTorch 2.0+)
        if hasattr(torch, 'compile') and config['infrastructure'].get('compile_model', False):
            print("⚡ Compiling model with torch.compile()...")
            self.model = torch.compile(
                self.model, 
                mode=config['infrastructure'].get('compile_mode', 'reduce-overhead')
            )
        
        # Setup mixed precision training
        self.scaler = GradScaler() if config['infrastructure']['mixed_precision'] else None
        
        # Setup data loaders with optimization
        self.train_loader = self._create_fast_dataloader('train')
        self.val_loader = self._create_fast_dataloader('val')
        
        # Setup optimizer with optimizations
        self.optimizer = self._create_fast_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Metrics
        self.metrics = MolecularMetrics()
        
        # Training state
        self.best_loss = float('inf')
        self.patience_counter = 0
        
    def _create_optimized_model(self):
        """Create model with performance optimizations"""
        print("📦 Loading optimized model...")
        
        # Use smaller configuration for speed
        model = End2EndMolecularGenerator(
            hidden_size=self.config['model']['hidden_size'],
            num_attention_heads=self.config['model']['num_attention_heads'],
            num_fusion_layers=self.config['model']['num_fusion_layers'],
            use_gradient_checkpointing=self.config['infrastructure']['gradient_checkpointing'],
            device=self.device
        )
        
        # Move to device and optimize memory layout
        model = model.to(self.device)
        
        # Use channels_last memory format for CNN components
        if self.config['optimization_strategies'].get('use_channels_last', False):
            model = model.to(memory_format=torch.channels_last)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📊 Total parameters: {total_params/1e6:.2f}M")
        print(f"📊 Trainable parameters: {trainable_params/1e6:.2f}M")
        
        return model
    
    def _create_fast_dataloader(self, split):
        """Create optimized dataloader"""
        print(f"📂 Loading {split} data...")
        
        # Load dataset
        data_path = self.config['data'][f'{split}_data']
        dataset = ScaffoldDataset(
            data_path=data_path,
            max_text_length=self.config['data']['max_text_length'],
            max_smiles_length=self.config['data']['max_smiles_length'],
            cache_data=self.config['optimization_strategies'].get('cache_dataset', True)
        )
        
        print(f"  ✓ {len(dataset)} samples loaded")
        
        # Create optimized dataloader
        batch_size = self.config['training']['batch_size'] if split == 'train' else \
                     self.config['training']['eval_batch_size']
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=self.config['infrastructure']['num_workers'],
            pin_memory=self.config['infrastructure']['pin_memory'],
            persistent_workers=self.config['infrastructure']['persistent_workers'],
            prefetch_factor=self.config['infrastructure'].get('prefetch_factor', 2),
            drop_last=(split == 'train')  # Drop last for consistent batch sizes
        )
        
        return loader
    
    def _create_fast_optimizer(self):
        """Create optimizer with layer-wise learning rates"""
        # Group parameters by component
        param_groups = [
            {'params': [], 'lr': self.config['training']['optimizer']['learning_rate']},
        ]
        
        base_lr = self.config['training']['optimizer']['learning_rate']
        lr_multipliers = self.config['training']['optimizer']['lr_multipliers']
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            # Assign to appropriate group based on name
            if 'molt5' in name.lower():
                lr = base_lr * lr_multipliers.get('molt5', 0.01)
            elif 'encoder' in name.lower():
                lr = base_lr * lr_multipliers.get('encoders', 0.1)
            elif 'fusion' in name.lower():
                lr = base_lr * lr_multipliers.get('fusion', 1.0)
            elif 'decoder' in name.lower():
                lr = base_lr * lr_multipliers.get('decoders', 1.0)
            elif 'adapter' in name.lower():
                lr = base_lr * lr_multipliers.get('adapters', 2.0)
            else:
                lr = base_lr
            
            # Find or create group
            found = False
            for group in param_groups:
                if group['lr'] == lr:
                    group['params'].append(param)
                    found = True
                    break
            
            if not found:
                param_groups.append({'params': [param], 'lr': lr})
        
        # Use fused optimizer if available
        if self.config['optimization_strategies'].get('use_fused_optimizers', True):
            try:
                from torch.optim import AdamW
                optimizer = AdamW(
                    param_groups,
                    weight_decay=self.config['training']['optimizer']['weight_decay'],
                    eps=self.config['training']['optimizer']['eps'],
                    betas=self.config['training']['optimizer']['betas'],
                    fused=True  # Use fused kernel
                )
                print("⚡ Using fused AdamW optimizer")
            except:
                optimizer = torch.optim.AdamW(
                    param_groups,
                    weight_decay=self.config['training']['optimizer']['weight_decay'],
                    eps=self.config['training']['optimizer']['eps'],
                    betas=self.config['training']['optimizer']['betas']
                )
        else:
            optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=self.config['training']['optimizer']['weight_decay']
            )
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        total_steps = len(self.train_loader) * self.config['training']['num_epochs']
        warmup_steps = self.config['training']['scheduler']['warmup_steps']
        
        scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        return scheduler
    
    def train_epoch(self, epoch):
        """Train for one epoch with optimizations"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        # Progress tracking
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Mixed precision training
            if self.scaler:
                with autocast():
                    outputs = self.model(**batch)
                    loss = outputs['loss']
                    
                    # Gradient accumulation
                    grad_acc_steps = self.config['training']['gradient_accumulation_steps']
                    loss = loss / grad_acc_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % grad_acc_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['training']['gradient_clip_norm']
                    )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            else:
                outputs = self.model(**batch)
                loss = outputs['loss']
                
                grad_acc_steps = self.config['training']['gradient_accumulation_steps']
                loss = loss / grad_acc_steps
                loss.backward()
                
                if (batch_idx + 1) % grad_acc_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip_norm']
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            
            total_loss += loss.item() * grad_acc_steps
            
            # Fast progress reporting
            if batch_idx % 50 == 0:
                elapsed = time.time() - start_time
                batches_per_sec = (batch_idx + 1) / elapsed
                eta = (num_batches - batch_idx - 1) / batches_per_sec
                
                print(f"\rEpoch {epoch} [{batch_idx}/{num_batches}] "
                      f"Loss: {loss.item():.4f} "
                      f"Speed: {batches_per_sec:.2f} batch/s "
                      f"ETA: {eta:.0f}s", end='')
        
        avg_loss = total_loss / num_batches
        elapsed = time.time() - start_time
        
        print(f"\n✅ Epoch {epoch} completed in {elapsed:.0f}s - Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def validate(self, epoch):
        """Fast validation"""
        # Skip validation for certain epochs
        skip_epochs = self.config['optimization_strategies'].get('skip_validation_epochs', [])
        if epoch in skip_epochs:
            print(f"⏭️  Skipping validation for epoch {epoch}")
            return None
        
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                if self.scaler:
                    with autocast():
                        outputs = self.model(**batch)
                        loss = outputs['loss']
                else:
                    outputs = self.model(**batch)
                    loss = outputs['loss']
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        print(f"📊 Validation Loss: {avg_loss:.4f}")
        
        # Early stopping check
        if self.config['optimization_strategies']['early_stopping']['enabled']:
            if avg_loss < self.best_loss - self.config['optimization_strategies']['early_stopping']['min_delta']:
                self.best_loss = avg_loss
                self.patience_counter = 0
                
                # Save best model
                if self.config['output']['save_best_model']:
                    self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
                
                if self.patience_counter >= self.config['optimization_strategies']['early_stopping']['patience']:
                    print(f"🛑 Early stopping triggered at epoch {epoch}")
                    return 'stop'
        
        return avg_loss
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config['output']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_name = 'best_model.pt' if is_best else f'checkpoint_epoch_{epoch}.pt'
        checkpoint_path = checkpoint_dir / checkpoint_name
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss
        }, checkpoint_path)
        
        print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*50)
        print("🚀 FAST TRAINING MODE - Target: 2-4 hours")
        print("="*50)
        
        total_start_time = time.time()
        
        for epoch in range(1, self.config['training']['num_epochs'] + 1):
            print(f"\n📅 Epoch {epoch}/{self.config['training']['num_epochs']}")
            
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation
            val_result = self.validate(epoch)
            
            if val_result == 'stop':
                break
            
            # Save checkpoint
            if epoch % self.config['training']['save_epochs'] == 0:
                self.save_checkpoint(epoch)
        
        # Training completed
        total_time = time.time() - total_start_time
        hours = total_time / 3600
        
        print("\n" + "="*50)
        print(f"✅ Training completed in {hours:.1f} hours!")
        print("="*50)
        
        # Save final model
        self.save_checkpoint(epoch, is_best=False)

def main():
    parser = argparse.ArgumentParser(description='Fast training for scaffold-based molecular generation')
    parser.add_argument('--config', type=str, default='configs/fast_training_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    args = parser.parse_args()
    
    # Set GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer and start training
    trainer = FastTrainer(config)
    trainer.train()

if __name__ == '__main__':
    main()