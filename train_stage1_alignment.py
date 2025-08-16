#!/usr/bin/env python3
"""
Stage 1: Multi-Modal Alignment Pre-training

This script implements the first stage of training focused on learning
cross-modal alignments through contrastive learning and matching tasks.
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

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scaffold_mol_gen.models.encoders import MultiModalEncoder
from scaffold_mol_gen.models.fusion_simplified import MultiModalFusionLayer
from scaffold_mol_gen.data.dataset import MultiModalDataset
from scaffold_mol_gen.training.loss_functions import ContrastiveLoss, MatchingLoss
from scaffold_mol_gen.utils.training_utils import AverageMeter, save_checkpoint, load_checkpoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('stage1_alignment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AlignmentModel(nn.Module):
    """
    Stage 1 Alignment Model
    
    Combines multi-modal encoders with GIT-Former fusion
    for cross-modal alignment learning.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Multi-modal encoders
        self.encoder = MultiModalEncoder(
            hidden_size=config['hidden_size'],
            use_scibert=config.get('use_scibert', False),
            freeze_backbones=config.get('freeze_encoders', True),
            device=config['device']
        )
        
        # GIT-Former fusion layer
        self.fusion = MultiModalFusionLayer(
            hidden_size=config['hidden_size'],
            num_heads=config.get('num_heads', 8),
            dropout=config.get('dropout', 0.1)
        )
        
        # Projection heads for contrastive learning
        self.projection_dim = config.get('projection_dim', 256)
        self.smiles_proj = nn.Linear(config['hidden_size'], self.projection_dim)
        self.graph_proj = nn.Linear(config['hidden_size'], self.projection_dim)
        self.image_proj = nn.Linear(config['hidden_size'], self.projection_dim)
        self.text_proj = nn.Linear(config['hidden_size'], self.projection_dim)
        
        # Temperature parameter for contrastive loss
        self.temperature = nn.Parameter(torch.tensor(config.get('temperature', 0.07)))
        
    def forward(self, batch: Dict) -> Dict:
        """
        Forward pass for alignment training
        
        Args:
            batch: Dictionary containing multi-modal inputs
            
        Returns:
            Dictionary with encoded features and projections
        """
        outputs = {}
        
        # Encode each modality if present
        if 'smiles' in batch:
            smiles_features = self.encoder.encode_smiles(batch['smiles'])
            outputs['smiles_features'] = smiles_features
            outputs['smiles_proj'] = self.smiles_proj(smiles_features)
            
        if 'graph' in batch:
            graph_features = self.encoder.encode_graph(batch['graph'])
            outputs['graph_features'] = graph_features
            outputs['graph_proj'] = self.graph_proj(graph_features)
            
        if 'image' in batch:
            image_features = self.encoder.encode_image(batch['image'])
            outputs['image_features'] = image_features
            outputs['image_proj'] = self.image_proj(image_features)
            
        if 'text' in batch:
            text_features = self.encoder.encode_text(batch['text'])
            outputs['text_features'] = text_features
            outputs['text_proj'] = self.text_proj(text_features)
        
        outputs['temperature'] = self.temperature
        
        return outputs


class Stage1Trainer:
    """
    Trainer for Stage 1 alignment pre-training
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['device'])
        
        # Create model
        self.model = AlignmentModel(config).to(self.device)
        
        # Loss functions
        self.contrastive_loss = ContrastiveLoss()
        self.matching_loss = MatchingLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs']
        )
        
        # Metrics
        self.train_metrics = {
            'loss': AverageMeter(),
            'contrastive_loss': AverageMeter(),
            'matching_loss': AverageMeter()
        }
        
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.train()
        self.train_metrics = {k: AverageMeter() for k in self.train_metrics}
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(batch)
            
            # Calculate losses
            total_loss = 0
            
            # Contrastive losses for different pairs
            if 'smiles_proj' in outputs and 'text_proj' in outputs:
                st_contrastive = self.contrastive_loss(
                    outputs['smiles_proj'], 
                    outputs['text_proj'],
                    outputs['temperature']
                )
                total_loss += st_contrastive * self.config['loss_weights']['st_contrastive']
                
            if 'graph_proj' in outputs and 'text_proj' in outputs:
                gt_contrastive = self.contrastive_loss(
                    outputs['graph_proj'],
                    outputs['text_proj'],
                    outputs['temperature']
                )
                total_loss += gt_contrastive * self.config['loss_weights']['gt_contrastive']
                
            if 'image_proj' in outputs and 'text_proj' in outputs:
                it_contrastive = self.contrastive_loss(
                    outputs['image_proj'],
                    outputs['text_proj'],
                    outputs['temperature']
                )
                total_loss += it_contrastive * self.config['loss_weights']['it_contrastive']
            
            # Matching losses
            if 'smiles_features' in outputs and 'text_features' in outputs:
                st_matching = self.matching_loss(
                    outputs['smiles_features'],
                    outputs['text_features'],
                    batch.get('st_labels')
                )
                total_loss += st_matching * self.config['loss_weights']['st_matching']
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Update metrics
            self.train_metrics['loss'].update(total_loss.item())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'temp': f'{outputs["temperature"].item():.3f}'
            })
            
        return {k: v.avg for k, v in self.train_metrics.items()}
    
    def validate(self, dataloader: DataLoader) -> Dict:
        """Validate the model"""
        self.model.eval()
        val_metrics = {k: AverageMeter() for k in self.train_metrics}
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validation'):
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch)
                
                # Calculate losses (same as training)
                total_loss = 0
                
                # Add contrastive and matching losses...
                # (Similar to training loop)
                
                val_metrics['loss'].update(total_loss.item())
        
        return {k: v.avg for k, v in val_metrics.items()}
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop"""
        logger.info("Starting Stage 1 Alignment Training")
        logger.info(f"Config: {self.config}")
        
        best_val_loss = float('inf')
        
        for epoch in range(1, self.config['num_epochs'] + 1):
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Logging
            logger.info(f"Epoch {epoch}/{self.config['num_epochs']}")
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            
            # Save checkpoint
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics['loss'],
                    self.config['output_dir'] / 'best_model.pt'
                )
                logger.info("Saved best model")
            
            # Regular checkpoint
            if epoch % self.config.get('save_every', 10) == 0:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics['loss'],
                    self.config['output_dir'] / f'checkpoint_epoch_{epoch}.pt'
                )
        
        logger.info("Stage 1 training completed!")


def main():
    parser = argparse.ArgumentParser(description='Stage 1: Alignment Pre-training')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='data/chebi20_mm/',
                       help='Path to multi-modal dataset')
    parser.add_argument('--output_dir', type=str, default='models/stage1/',
                       help='Output directory for models')
    
    # Model arguments
    parser.add_argument('--hidden_size', type=int, default=768,
                       help='Hidden size for encoders')
    parser.add_argument('--projection_dim', type=int, default=256,
                       help='Projection dimension for contrastive learning')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='Temperature for contrastive loss')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    # Loss weights
    parser.add_argument('--st_contrastive', type=float, default=1.0,
                       help='Weight for SMILES-Text contrastive loss')
    parser.add_argument('--gt_contrastive', type=float, default=1.0,
                       help='Weight for Graph-Text contrastive loss')
    parser.add_argument('--it_contrastive', type=float, default=1.0,
                       help='Weight for Image-Text contrastive loss')
    parser.add_argument('--st_matching', type=float, default=1.0,
                       help='Weight for SMILES-Text matching loss')
    
    args = parser.parse_args()
    
    # Create config
    config = {
        'data_path': Path(args.data_path),
        'output_dir': Path(args.output_dir),
        'hidden_size': args.hidden_size,
        'projection_dim': args.projection_dim,
        'num_heads': args.num_heads,
        'temperature': args.temperature,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'device': args.device,
        'loss_weights': {
            'st_contrastive': args.st_contrastive,
            'gt_contrastive': args.gt_contrastive,
            'it_contrastive': args.it_contrastive,
            'st_matching': args.st_matching
        },
        'freeze_encoders': True,
        'use_scibert': False
    }
    
    # Create output directory
    config['output_dir'].mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(config['output_dir'] / 'config.json', 'w') as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in config.items()}, 
                 f, indent=2)
    
    # Create data loaders
    train_dataset = MultiModalDataset(
        config['data_path'] / 'train.csv',
        modalities=['smiles', 'graph', 'image', 'text']
    )
    val_dataset = MultiModalDataset(
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
    trainer = Stage1Trainer(config)
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()