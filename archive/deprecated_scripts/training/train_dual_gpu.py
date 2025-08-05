#!/usr/bin/env python3
"""
Dual GPU Training Script - Optimized for 2x RTX 4090 or 32GB vGPU
Target: 1-2 hours training with 90%+ model performance
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_cosine_schedule_with_warmup
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from scaffold_mol_gen.models.end2end_model import End2EndMolecularGenerator
from scaffold_mol_gen.data.dataset import ScaffoldDataset
from scaffold_mol_gen.training.metrics import MolecularMetrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DualGPUTrainer:
    """High-performance trainer for dual GPU setup"""
    
    def __init__(self, config, rank=0, world_size=1):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank}')
        
        # Set device
        torch.cuda.set_device(self.device)
        
        if rank == 0:
            print(f"🚀 Dual GPU Training Mode")
            print(f"📊 Using {world_size} GPUs")
            print(f"🎯 Target: 1-2 hours training")
            print("="*50)
        
        # Enable all optimizations
        self._enable_optimizations()
        
        # Initialize model
        self.model = self._create_model()
        
        # Setup distributed model
        if world_size > 1:
            self.model = DDP(
                self.model, 
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=False,
                gradient_as_bucket_view=True
            )
        
        # Mixed precision setup with bfloat16 for 4090
        self.use_amp = config['infrastructure']['mixed_precision']
        if self.use_amp:
            # Use bfloat16 if available (better for 4090)
            if config['infrastructure'].get('mixed_precision_dtype') == 'bfloat16' and \
               torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.bfloat16
                if rank == 0:
                    print("⚡ Using bfloat16 mixed precision")
            else:
                self.amp_dtype = torch.float16
                if rank == 0:
                    print("⚡ Using float16 mixed precision")
            
            self.scaler = GradScaler()
        
        # Data loaders
        self.train_loader = self._create_dataloader('train')
        self.val_loader = self._create_dataloader('val')
        
        # Optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Metrics
        self.metrics = MolecularMetrics() if rank == 0 else None
        
        # Training state
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.start_time = time.time()
    
    def _enable_optimizations(self):
        """Enable all CUDA optimizations for RTX 4090"""
        if torch.cuda.is_available():
            # Enable TensorFloat-32 for A100/4090
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable cuDNN autotuner
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            # Set CUDA memory allocator
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
            
            if self.rank == 0:
                print("✅ CUDA optimizations enabled")
                print(f"  - TF32: Enabled")
                print(f"  - cuDNN Benchmark: Enabled")
                print(f"  - Memory Allocator: Optimized")
    
    def _create_model(self):
        """Create optimized model"""
        if self.rank == 0:
            print("📦 Loading model...")
        
        model = End2EndMolecularGenerator(
            hidden_size=self.config['model']['hidden_size'],
            num_attention_heads=self.config['model']['num_attention_heads'],
            num_fusion_layers=self.config['model']['num_fusion_layers'],
            use_gradient_checkpointing=False,  # Not needed with 24GB
            device=self.device
        )
        
        model = model.to(self.device)
        
        # Compile model with torch.compile for RTX 4090
        if hasattr(torch, 'compile') and self.config['infrastructure'].get('compile_model'):
            if self.rank == 0:
                print("🔧 Compiling model with torch.compile()...")
            model = torch.compile(
                model,
                mode=self.config['infrastructure'].get('compile_mode', 'max-autotune'),
                backend=self.config['infrastructure'].get('compile_backend', 'inductor')
            )
        
        # Use channels_last memory format
        if self.config['infrastructure'].get('use_channels_last'):
            model = model.to(memory_format=torch.channels_last)
        
        if self.rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"📊 Model size: {total_params/1e6:.2f}M total, {trainable_params/1e6:.2f}M trainable")
        
        return model
    
    def _create_dataloader(self, split):
        """Create distributed dataloader"""
        data_path = self.config['data'][f'{split}_data']
        
        # Create dataset
        dataset = ScaffoldDataset(
            data_path=data_path,
            max_text_length=self.config['data']['max_text_length'],
            max_smiles_length=self.config['data']['max_smiles_length'],
            cache_data=True
        )
        
        # Batch size per GPU
        batch_size = self.config['training']['batch_size']
        if split == 'val':
            batch_size = self.config['training']['eval_batch_size']
        
        # Create sampler for distributed training
        sampler = None
        if self.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=(split == 'train')
            )
        
        # Create dataloader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train' and sampler is None),
            sampler=sampler,
            num_workers=self.config['infrastructure']['num_workers'] // self.world_size,
            pin_memory=self.config['infrastructure']['pin_memory'],
            persistent_workers=self.config['infrastructure']['persistent_workers'],
            prefetch_factor=self.config['infrastructure'].get('prefetch_factor', 2),
            drop_last=(split == 'train')
        )
        
        if self.rank == 0:
            print(f"📂 {split} dataset: {len(dataset)} samples, {len(loader)} batches")
        
        return loader
    
    def _create_optimizer(self):
        """Create optimizer with differential learning rates"""
        # Group parameters
        param_groups = []
        base_lr = self.config['training']['optimizer']['learning_rate']
        lr_multipliers = self.config['training']['optimizer']['lr_multipliers']
        
        # Get model (handle DDP wrapper)
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Determine learning rate
            if 'molt5' in name.lower():
                lr = base_lr * lr_multipliers.get('molt5', 0.05)
                group_name = 'molt5'
            elif 'encoder' in name.lower():
                lr = base_lr * lr_multipliers.get('encoders', 0.2)
                group_name = 'encoders'
            elif 'fusion' in name.lower():
                lr = base_lr * lr_multipliers.get('fusion', 1.0)
                group_name = 'fusion'
            elif 'decoder' in name.lower():
                lr = base_lr * lr_multipliers.get('decoders', 1.0)
                group_name = 'decoders'
            elif 'adapter' in name.lower():
                lr = base_lr * lr_multipliers.get('adapters', 2.0)
                group_name = 'adapters'
            else:
                lr = base_lr
                group_name = 'other'
            
            # Add to appropriate group
            found = False
            for group in param_groups:
                if group['name'] == group_name:
                    group['params'].append(param)
                    found = True
                    break
            
            if not found:
                param_groups.append({
                    'params': [param],
                    'lr': lr,
                    'name': group_name
                })
        
        # Create optimizer with fused kernels
        if self.config['infrastructure'].get('use_fused_adam', True):
            try:
                optimizer = torch.optim.AdamW(
                    param_groups,
                    weight_decay=self.config['training']['optimizer']['weight_decay'],
                    eps=self.config['training']['optimizer']['eps'],
                    betas=self.config['training']['optimizer']['betas'],
                    fused=True
                )
                if self.rank == 0:
                    print("⚡ Using fused AdamW optimizer")
            except:
                optimizer = torch.optim.AdamW(
                    param_groups,
                    weight_decay=self.config['training']['optimizer']['weight_decay']
                )
        else:
            optimizer = torch.optim.AdamW(param_groups)
        
        return optimizer
    
    def _create_scheduler(self):
        """Create OneCycle learning rate scheduler"""
        total_steps = len(self.train_loader) * self.config['training']['num_epochs']
        
        # Use OneCycle for fast convergence
        if self.config['training']['scheduler']['type'] == 'onecycle':
            from torch.optim.lr_scheduler import OneCycleLR
            scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config['training']['scheduler']['max_lr'],
                total_steps=total_steps,
                pct_start=self.config['training']['scheduler'].get('pct_start', 0.1),
                anneal_strategy=self.config['training']['scheduler'].get('anneal_strategy', 'cos')
            )
        else:
            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config['training']['scheduler'].get('warmup_steps', 100),
                num_training_steps=total_steps
            )
        
        return scheduler
    
    def train_epoch(self, epoch):
        """Train one epoch with distributed training"""
        self.model.train()
        
        # Set epoch for distributed sampler
        if self.world_size > 1 and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)
        
        total_loss = 0
        num_batches = len(self.train_loader)
        epoch_start = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Mixed precision training
            if self.use_amp:
                with autocast(dtype=self.amp_dtype):
                    outputs = self.model(**batch)
                    loss = outputs['loss']
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip_norm']
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)  # More efficient
            else:
                outputs = self.model(**batch)
                loss = outputs['loss']
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip_norm']
                )
                
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            
            self.scheduler.step()
            total_loss += loss.item()
            
            # Progress reporting (only rank 0)
            if self.rank == 0 and batch_idx % 20 == 0:
                elapsed = time.time() - epoch_start
                samples_per_sec = (batch_idx + 1) * self.config['training']['batch_size'] * self.world_size / elapsed
                gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
                
                print(f"\rEpoch {epoch} [{batch_idx}/{num_batches}] "
                      f"Loss: {loss.item():.4f} "
                      f"Speed: {samples_per_sec:.0f} samples/s "
                      f"GPU: {gpu_mem:.1f}GB "
                      f"LR: {self.scheduler.get_last_lr()[0]:.2e}", end='')
            
            # Clear cache periodically
            if batch_idx % self.config['infrastructure'].get('empty_cache_freq', 100) == 0:
                torch.cuda.empty_cache()
        
        # Gather loss from all GPUs
        avg_loss = total_loss / num_batches
        if self.world_size > 1:
            avg_loss_tensor = torch.tensor(avg_loss).to(self.device)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = avg_loss_tensor.item()
        
        if self.rank == 0:
            elapsed = time.time() - epoch_start
            total_samples = len(self.train_loader.dataset)
            print(f"\n✅ Epoch {epoch}: Loss={avg_loss:.4f}, "
                  f"Time={elapsed:.0f}s, "
                  f"Speed={total_samples/elapsed:.0f} samples/s")
        
        return avg_loss
    
    def validate(self, epoch):
        """Validation with distributed evaluation"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                if self.use_amp:
                    with autocast(dtype=self.amp_dtype):
                        outputs = self.model(**batch)
                        loss = outputs['loss']
                else:
                    outputs = self.model(**batch)
                    loss = outputs['loss']
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Gather from all GPUs
        if self.world_size > 1:
            avg_loss_tensor = torch.tensor(avg_loss).to(self.device)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = avg_loss_tensor.item()
        
        if self.rank == 0:
            print(f"📊 Validation Loss: {avg_loss:.4f}")
            
            # Early stopping check
            if self.config['optimization_strategies']['early_stopping']['enabled']:
                if avg_loss < self.best_loss - self.config['optimization_strategies']['early_stopping']['min_delta']:
                    self.best_loss = avg_loss
                    self.patience_counter = 0
                    self.save_checkpoint(epoch, is_best=True)
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config['optimization_strategies']['early_stopping']['patience']:
                        return 'stop'
        
        return avg_loss
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint (only rank 0)"""
        if self.rank != 0:
            return
        
        checkpoint_dir = Path(self.config['output']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model without DDP wrapper
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        # Optionally save optimizer state
        if self.config['output'].get('save_optimizer_state', False):
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        checkpoint_name = 'best_model.pt' if is_best else f'checkpoint_epoch_{epoch}.pt'
        checkpoint_path = checkpoint_dir / checkpoint_name
        
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    def train(self):
        """Main training loop"""
        if self.rank == 0:
            print("\n" + "="*60)
            print("🚀 DUAL GPU TRAINING - RTX 4090 Optimized")
            print(f"📊 Total batch size: {self.config['training']['batch_size'] * self.world_size}")
            print(f"🎯 Target: 1-2 hours for 90%+ performance")
            print("="*60 + "\n")
        
        for epoch in range(1, self.config['training']['num_epochs'] + 1):
            if self.rank == 0:
                print(f"\n📅 Epoch {epoch}/{self.config['training']['num_epochs']}")
            
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation
            if epoch % self.config['training']['eval_epochs'] == 0:
                val_result = self.validate(epoch)
                if val_result == 'stop':
                    if self.rank == 0:
                        print("🛑 Early stopping triggered")
                    break
            
            # Save checkpoint
            if self.rank == 0 and epoch % self.config['training']['save_epochs'] == 0:
                self.save_checkpoint(epoch)
        
        # Training completed
        if self.rank == 0:
            total_time = time.time() - self.start_time
            hours = total_time / 3600
            print("\n" + "="*60)
            print(f"✅ Training completed in {hours:.2f} hours!")
            print(f"🎯 Final loss: {train_loss:.4f}")
            print(f"⚡ Average speed: {len(self.train_loader.dataset)*self.config['training']['num_epochs']/total_time:.0f} samples/s")
            print("="*60)
            
            # Save final model
            self.save_checkpoint(epoch, is_best=False)

def setup_distributed():
    """Setup distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
    
    return rank, world_size, local_rank

def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description='Dual GPU training for molecular generation')
    parser.add_argument('--config', type=str, default='configs/dual_4090_config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer
    trainer = DualGPUTrainer(config, rank, world_size)
    
    try:
        # Start training
        trainer.train()
    finally:
        # Cleanup
        cleanup_distributed()

if __name__ == '__main__':
    main()