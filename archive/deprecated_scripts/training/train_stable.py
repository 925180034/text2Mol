#!/usr/bin/env python3
"""
Stable training script with progressive training and NaN prevention.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
import yaml
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from scaffold_mol_gen.models.stable_model import StableMolecularGenerator
from scaffold_mol_gen.data.scaffold_dataset import ScaffoldDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StableTrainer:
    """Trainer with stability features and NaN prevention"""
    
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Training on device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # Initialize components
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.nan_count = 0
        self.patience_counter = 0
        
        # Initialize everything
        self._initialize()
    
    def _load_config(self, config_path):
        """Load configuration with defaults"""
        default_config = {
            'training': {
                'batch_size': 2,  # Very small for stability
                'learning_rate': 1e-6,  # Very low
                'num_epochs': 50,
                'gradient_clip': 0.1,
                'warmup_epochs': 5,
                'patience': 10,
                'checkpoint_dir': 'checkpoints/',
                'save_every': 5
            },
            'model': {
                'freeze_molt5_encoder': True,
                'freeze_molt5_decoder_layers': 20,
                'use_simple_fusion': True
            },
            'data': {
                'train_path': 'Datasets/train.csv',
                'val_path': 'Datasets/validation.csv',
                'max_text_length': 256,
                'max_smiles_length': 128,
                'num_workers': 2
            }
        }
        
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            # Merge configs
            for key in user_config:
                if key in default_config:
                    default_config[key].update(user_config[key])
        
        return default_config
    
    def _initialize(self):
        """Initialize model, data, and optimizer"""
        # Create model
        logger.info("Creating model...")
        self.model = StableMolecularGenerator(
            freeze_molt5_encoder=self.config['model']['freeze_molt5_encoder'],
            freeze_molt5_decoder_layers=self.config['model']['freeze_molt5_decoder_layers'],
            use_simple_fusion=self.config['model']['use_simple_fusion']
        ).to(self.device)
        
        # Create datasets
        logger.info("Loading datasets...")
        train_dataset = ScaffoldDataset(
            data_path=self.config['data']['train_path'],
            max_text_length=self.config['data']['max_text_length'],
            max_smiles_length=self.config['data']['max_smiles_length'],
            cache_data=False  # Don't cache to save memory
        )
        
        val_dataset = ScaffoldDataset(
            data_path=self.config['data']['val_path'],
            max_text_length=self.config['data']['max_text_length'],
            max_smiles_length=self.config['data']['max_smiles_length'],
            cache_data=False
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'] * 2,
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True
        )
        
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Val batches: {len(self.val_loader)}")
        
        # Create optimizer with very low learning rate
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=1e-5,
            eps=1e-8
        )
        
        # Create scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-8
        )
        
        # Create checkpoint directory
        Path(self.config['training']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch with NaN detection"""
        self.model.train()
        total_loss = 0
        valid_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            try:
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs['loss']
                
                # Check for NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    self.nan_count += 1
                    logger.warning(f"NaN/Inf detected at batch {batch_idx}, skipping...")
                    
                    # Reset model if too many NaNs
                    if self.nan_count > 10:
                        logger.error("Too many NaNs, stopping training")
                        return None
                    continue
                
                # Backward pass with gradient clipping
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
                
                # Check for NaN gradients
                has_nan_grad = False
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            logger.warning(f"NaN/Inf gradient in {name}")
                            has_nan_grad = True
                            break
                
                if has_nan_grad:
                    self.nan_count += 1
                    continue
                
                # Optimizer step
                self.optimizer.step()
                
                # Track loss
                total_loss += loss.item()
                valid_batches += 1
                
                # Progress
                if batch_idx % 100 == 0:
                    avg_loss = total_loss / max(valid_batches, 1)
                    logger.info(f"Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}, Avg: {avg_loss:.4f}")
                
                # Clear cache periodically
                if batch_idx % 500 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                continue
        
        if valid_batches == 0:
            logger.error("No valid batches in epoch!")
            return None
        
        avg_loss = total_loss / valid_batches
        return avg_loss
    
    def validate(self):
        """Validation with NaN detection"""
        self.model.eval()
        total_loss = 0
        valid_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                try:
                    outputs = self.model(**batch)
                    loss = outputs['loss']
                    
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        total_loss += loss.item()
                        valid_batches += 1
                except:
                    continue
        
        if valid_batches == 0:
            return float('inf')
        
        avg_loss = total_loss / valid_batches
        return avg_loss
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = Path(self.config['training']['checkpoint_dir']) / f'checkpoint_epoch_{self.current_epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = Path(self.config['training']['checkpoint_dir']) / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
    
    def train(self):
        """Main training loop with progressive training"""
        logger.info("="*60)
        logger.info("Starting Stable Training")
        logger.info(f"Epochs: {self.config['training']['num_epochs']}")
        logger.info(f"Batch size: {self.config['training']['batch_size']}")
        logger.info(f"Learning rate: {self.config['training']['learning_rate']}")
        logger.info("="*60)
        
        # Progressive training: start with very low learning rate
        warmup_epochs = self.config['training']['warmup_epochs']
        base_lr = self.config['training']['learning_rate']
        
        for epoch in range(1, self.config['training']['num_epochs'] + 1):
            self.current_epoch = epoch
            
            # Warmup learning rate
            if epoch <= warmup_epochs:
                warmup_lr = base_lr * (epoch / warmup_epochs)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                logger.info(f"Warmup epoch {epoch}/{warmup_epochs}, LR: {warmup_lr:.2e}")
            
            logger.info(f"\n{'='*20} Epoch {epoch}/{self.config['training']['num_epochs']} {'='*20}")
            
            # Training
            train_loss = self.train_epoch()
            
            if train_loss is None:
                logger.error("Training failed, stopping...")
                break
            
            # Validation
            val_loss = self.validate()
            
            # Scheduler step
            if epoch > warmup_epochs:
                self.scheduler.step(val_loss)
            
            logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(is_best=True)
            else:
                self.patience_counter += 1
            
            # Save regular checkpoint
            if epoch % self.config['training']['save_every'] == 0:
                self.save_checkpoint()
            
            # Early stopping
            if self.patience_counter >= self.config['training']['patience']:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
            
            # Reset NaN counter each epoch
            self.nan_count = 0
        
        logger.info("="*60)
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {self.best_loss:.4f}")
        logger.info("="*60)

def main():
    parser = argparse.ArgumentParser(description='Stable molecular generation training')
    parser.add_argument('--config', type=str, default='configs/stable_config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    # Create trainer and start training
    trainer = StableTrainer(args.config)
    trainer.train()

if __name__ == '__main__':
    main()