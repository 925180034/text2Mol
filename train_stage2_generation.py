#!/usr/bin/env python3
"""
Stage 2: Instruction-Guided Generative Fine-tuning

This script implements the second stage of training focused on
instruction-conditioned generation across all 9 modality combinations.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time
from datetime import datetime
import json
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple, Optional
import random

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scaffold_mol_gen.models.end2end_model import End2EndMolecularGenerator
from scaffold_mol_gen.data.dataset import GenerationDataset
from scaffold_mol_gen.evaluation.comprehensive_metrics import ComprehensiveMetrics
from scaffold_mol_gen.utils.training_utils import AverageMeter, save_checkpoint, load_checkpoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('stage2_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class InstructionGenerator(nn.Module):
    """
    Stage 2 Generation Model with Instruction Conditioning
    
    Extends the End2End model with instruction tokens for
    controlling output modality.
    """
    
    def __init__(self, config: Dict, stage1_checkpoint: Optional[str] = None):
        super().__init__()
        
        # Base model
        self.model = End2EndMolecularGenerator(
            hidden_size=config['hidden_size'],
            molt5_path=config.get('molt5_path', 'laituan245/molt5-large-caption2smiles'),
            use_scibert=config.get('use_scibert', False),
            freeze_encoders=config.get('freeze_encoders', True),
            freeze_molt5=config.get('freeze_molt5', False),
            device=config['device']
        )
        
        # Load Stage 1 weights if provided
        if stage1_checkpoint:
            self.load_stage1_weights(stage1_checkpoint)
        
        # Instruction tokens
        self.instruction_tokens = {
            'smiles': '[GEN_SMILES]',
            'graph': '[GEN_GRAPH]',
            'image': '[GEN_IMAGE]'
        }
        
        # Task embeddings
        self.task_embeddings = nn.Embedding(3, config['hidden_size'])
        
    def load_stage1_weights(self, checkpoint_path: str):
        """Load pre-trained weights from Stage 1"""
        logger.info(f"Loading Stage 1 weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load encoder and fusion weights
        encoder_state = {}
        fusion_state = {}
        
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith('encoder.'):
                encoder_state[k.replace('encoder.', '')] = v
            elif k.startswith('fusion.'):
                fusion_state[k.replace('fusion.', '')] = v
        
        self.model.encoder.load_state_dict(encoder_state, strict=False)
        self.model.fusion.load_state_dict(fusion_state, strict=False)
        
        logger.info("Stage 1 weights loaded successfully")
    
    def forward(self, batch: Dict) -> Dict:
        """
        Forward pass with instruction conditioning
        
        Args:
            batch: Dictionary containing inputs and target modality
            
        Returns:
            Dictionary with generation outputs
        """
        # Get task embedding based on output modality
        output_modality = batch['output_modality']
        task_id = ['smiles', 'graph', 'image'].index(output_modality)
        task_emb = self.task_embeddings(torch.tensor(task_id).to(batch['device']))
        
        # Forward through base model
        outputs = self.model(
            scaffold_data=batch['scaffold_data'],
            text_data=batch['text_data'],
            scaffold_modality=batch['scaffold_modality'],
            target_smiles=batch.get('target'),
            output_modality=output_modality
        )
        
        # Add task embedding to outputs
        outputs['task_embedding'] = task_emb
        
        return outputs


class Stage2Trainer:
    """
    Trainer for Stage 2 generation fine-tuning
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['device'])
        
        # Create model
        self.model = InstructionGenerator(
            config, 
            stage1_checkpoint=config.get('stage1_checkpoint')
        ).to(self.device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Optimizer with different learning rates
        encoder_params = []
        decoder_params = []
        
        for name, param in self.model.named_parameters():
            if 'encoder' in name:
                encoder_params.append(param)
            else:
                decoder_params.append(param)
        
        self.optimizer = optim.AdamW([
            {'params': encoder_params, 'lr': config.get('encoder_lr', 1e-6)},
            {'params': decoder_params, 'lr': config['learning_rate']}
        ], weight_decay=config.get('weight_decay', 0.01))
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[config.get('encoder_lr', 1e-6), config['learning_rate']],
            epochs=config['num_epochs'],
            steps_per_epoch=config.get('steps_per_epoch', 1000)
        )
        
        # Metrics calculator
        self.metrics_calculator = ComprehensiveMetrics()
        
        # Training metrics
        self.train_metrics = {
            'loss': AverageMeter(),
            'accuracy': AverageMeter()
        }
        
        # Define all 9 task combinations
        self.task_combinations = [
            ('smiles', 'smiles'), ('smiles', 'graph'), ('smiles', 'image'),
            ('graph', 'smiles'),  ('graph', 'graph'),  ('graph', 'image'),
            ('image', 'smiles'),  ('image', 'graph'),  ('image', 'image')
        ]
    
    def create_mixed_batch(self, batch: Dict) -> List[Dict]:
        """
        Create a mixed batch with different task combinations
        
        Args:
            batch: Original batch data
            
        Returns:
            List of task-specific batches
        """
        mixed_batches = []
        batch_size = len(batch['scaffold'])
        
        # Randomly assign tasks to each sample
        for i in range(batch_size):
            # Randomly select a task combination
            input_mod, output_mod = random.choice(self.task_combinations)
            
            task_batch = {
                'scaffold_data': batch['scaffold'][i],
                'text_data': batch['text'][i],
                'scaffold_modality': input_mod,
                'output_modality': output_mod,
                'target': batch['target'][i],
                'device': self.device
            }
            
            mixed_batches.append(task_batch)
        
        return mixed_batches
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.train()
        self.train_metrics = {k: AverageMeter() for k in self.train_metrics}
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            # Create mixed batch with different tasks
            mixed_batches = self.create_mixed_batch(batch)
            
            total_loss = 0
            total_accuracy = 0
            
            # Process each task in the mixed batch
            for task_batch in mixed_batches:
                # Forward pass
                outputs = self.model(task_batch)
                
                # Calculate loss based on output modality
                if task_batch['output_modality'] == 'smiles':
                    if 'loss' in outputs:
                        loss = outputs['loss']
                    else:
                        # Calculate cross-entropy loss for SMILES generation
                        logits = outputs.get('logits')
                        target = task_batch['target']
                        if logits is not None and target is not None:
                            loss = self.criterion(
                                logits.view(-1, logits.size(-1)),
                                target.view(-1)
                            )
                        else:
                            loss = torch.tensor(0.0).to(self.device)
                else:
                    # For graph and image outputs, use reconstruction loss
                    loss = outputs.get('loss', torch.tensor(0.0).to(self.device))
                
                total_loss += loss
            
            # Average loss over mixed batch
            total_loss = total_loss / len(mixed_batches)
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            self.train_metrics['loss'].update(total_loss.item())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
        
        return {k: v.avg for k, v in self.train_metrics.items()}
    
    def validate(self, dataloader: DataLoader) -> Dict:
        """Validate the model"""
        self.model.eval()
        val_metrics = {
            'loss': AverageMeter(),
            'validity': AverageMeter(),
            'uniqueness': AverageMeter(),
            'novelty': AverageMeter()
        }
        
        all_generated = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validation'):
                # Process only SMILES output for validation metrics
                batch_outputs = []
                
                for i in range(len(batch['scaffold'])):
                    task_batch = {
                        'scaffold_data': batch['scaffold'][i],
                        'text_data': batch['text'][i],
                        'scaffold_modality': 'smiles',
                        'output_modality': 'smiles',
                        'target': batch['target'][i],
                        'device': self.device
                    }
                    
                    outputs = self.model(task_batch)
                    
                    if 'generated_smiles' in outputs:
                        all_generated.append(outputs['generated_smiles'])
                        all_targets.append(batch['target'][i])
                    
                    # Update loss
                    if 'loss' in outputs:
                        val_metrics['loss'].update(outputs['loss'].item())
        
        # Calculate molecular metrics
        if all_generated:
            metrics = self.metrics_calculator.calculate_all_metrics(
                all_generated, all_targets
            )
            val_metrics['validity'].update(metrics['validity'])
            val_metrics['uniqueness'].update(metrics['uniqueness'])
            val_metrics['novelty'].update(metrics['novelty'])
        
        return {k: v.avg for k, v in val_metrics.items()}
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop"""
        logger.info("Starting Stage 2 Generation Training")
        logger.info(f"Config: {self.config}")
        logger.info(f"Training on {len(self.task_combinations)} task combinations")
        
        best_val_loss = float('inf')
        best_validity = 0.0
        
        for epoch in range(1, self.config['num_epochs'] + 1):
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Logging
            logger.info(f"Epoch {epoch}/{self.config['num_epochs']}")
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"Val Validity: {val_metrics['validity']:.2%}")
            logger.info(f"Val Uniqueness: {val_metrics['uniqueness']:.2%}")
            logger.info(f"Val Novelty: {val_metrics['novelty']:.2%}")
            
            # Save best model based on validity
            if val_metrics['validity'] > best_validity:
                best_validity = val_metrics['validity']
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics,
                    self.config['output_dir'] / 'best_model.pt'
                )
                logger.info(f"Saved best model with validity: {best_validity:.2%}")
            
            # Regular checkpoint
            if epoch % self.config.get('save_every', 10) == 0:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics,
                    self.config['output_dir'] / f'checkpoint_epoch_{epoch}.pt'
                )
        
        logger.info("Stage 2 training completed!")
        logger.info(f"Best validation validity: {best_validity:.2%}")


def main():
    parser = argparse.ArgumentParser(description='Stage 2: Generation Fine-tuning')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='data/chebi20_mm/',
                       help='Path to multi-modal dataset')
    parser.add_argument('--output_dir', type=str, default='models/stage2/',
                       help='Output directory for models')
    parser.add_argument('--stage1_checkpoint', type=str, 
                       default='models/stage1/best_model.pt',
                       help='Path to Stage 1 checkpoint')
    
    # Model arguments
    parser.add_argument('--hidden_size', type=int, default=768,
                       help='Hidden size for model')
    parser.add_argument('--molt5_path', type=str,
                       default='laituan245/molt5-large-caption2smiles',
                       help='Path to MolT5 model')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Learning rate for decoder')
    parser.add_argument('--encoder_lr', type=float, default=1e-6,
                       help='Learning rate for encoder fine-tuning')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--freeze_encoders', action='store_true',
                       help='Freeze encoder weights initially')
    
    args = parser.parse_args()
    
    # Create config
    config = {
        'data_path': Path(args.data_path),
        'output_dir': Path(args.output_dir),
        'stage1_checkpoint': args.stage1_checkpoint,
        'hidden_size': args.hidden_size,
        'molt5_path': args.molt5_path,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'encoder_lr': args.encoder_lr,
        'weight_decay': args.weight_decay,
        'device': args.device,
        'freeze_encoders': args.freeze_encoders,
        'freeze_molt5': False,
        'use_scibert': False,
        'save_every': 10,
        'steps_per_epoch': 1000  # Adjust based on dataset size
    }
    
    # Create output directory
    config['output_dir'].mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(config['output_dir'] / 'config.json', 'w') as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in config.items()}, 
                 f, indent=2)
    
    # Create data loaders
    train_dataset = GenerationDataset(
        config['data_path'] / 'train.csv',
        modalities=['smiles', 'graph', 'image', 'text']
    )
    val_dataset = GenerationDataset(
        config['data_path'] / 'val.csv',
        modalities=['smiles', 'graph', 'image', 'text']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create trainer and train
    trainer = Stage2Trainer(config)
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()