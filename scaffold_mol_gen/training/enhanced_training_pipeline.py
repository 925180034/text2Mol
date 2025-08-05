#!/usr/bin/env python3
"""
Enhanced Training Pipeline for Scaffold-based Molecular Generation.
Implements advanced training strategies, curriculum learning, and optimization techniques
specifically designed for the MolT5-Large Caption2SMILES model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import yaml
import logging
import json
import time
import math
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for enhanced training pipeline."""
    
    # Model configuration
    model_path: str = "models/MolT5-Large-Caption2SMILES"
    
    # Data configuration
    train_data: str = "Datasets/train.csv"
    val_data: str = "Datasets/validation.csv"
    max_text_length: int = 256
    max_smiles_length: int = 128
    
    # Training configuration
    num_epochs: int = 10
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Advanced training features
    use_curriculum_learning: bool = True
    use_molecular_metrics: bool = True
    use_scaffold_conditioning: bool = True
    use_adversarial_training: bool = False
    
    # Evaluation configuration
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    
    # Output configuration
    output_dir: str = "/root/autodl-tmp/text2Mol-outputs/enhanced_training"
    save_best_model: bool = True
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class MolecularDataset(Dataset):
    """Enhanced dataset with curriculum learning and molecular metrics."""
    
    def __init__(self, data_path: str, tokenizer, config: TrainingConfig, is_training: bool = True):
        self.tokenizer = tokenizer
        self.config = config
        self.is_training = is_training
        
        # Load and process data
        df = pd.read_csv(data_path)
        logger.info(f"Loading {len(df)} samples from {data_path}")
        
        self.data = []
        self.difficulty_scores = []
        
        valid_count = 0
        for idx, row in df.iterrows():
            smiles = row['SMILES']
            description = row['description']
            
            # Validate SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            # Calculate difficulty score for curriculum learning
            difficulty = self._calculate_difficulty(smiles, description, mol)
            
            # Create training examples with multiple prompt formats
            examples = self._create_training_examples(description, smiles)
            
            for example in examples:
                self.data.append(example)
                self.difficulty_scores.append(difficulty)
            
            valid_count += 1
            if valid_count >= 10000:  # Limit for training efficiency
                break
        
        logger.info(f"Created {len(self.data)} training examples from {valid_count} molecules")
        
        # Initialize curriculum learning
        if self.is_training and config.use_curriculum_learning:
            self._initialize_curriculum()
    
    def _calculate_difficulty(self, smiles: str, description: str, mol) -> float:
        """Calculate difficulty score for curriculum learning."""
        
        difficulty = 0.0
        
        # Molecular complexity
        try:
            num_atoms = mol.GetNumAtoms()
            num_rings = rdMolDescriptors.CalcNumRings(mol)
            molecular_weight = Descriptors.MolWt(mol)
            complexity = Descriptors.BertzCT(mol)
            
            # Normalize complexity factors
            difficulty += min(num_atoms / 50.0, 1.0) * 0.2  # Atom count
            difficulty += min(num_rings / 5.0, 1.0) * 0.2   # Ring count
            difficulty += min(molecular_weight / 500.0, 1.0) * 0.2  # Molecular weight
            difficulty += min(complexity / 1000.0, 1.0) * 0.2  # Bertz complexity
            
        except Exception:
            difficulty += 0.5  # Default for calculation errors
        
        # Text complexity
        text_complexity = len(description.split()) / 50.0  # Normalize by typical length
        difficulty += min(text_complexity, 1.0) * 0.2
        
        return min(difficulty, 1.0)  # Cap at 1.0
    
    def _create_training_examples(self, description: str, smiles: str) -> List[Dict]:
        """Create multiple training examples with different prompt formats."""
        
        examples = []
        
        # Basic caption format (primary)
        examples.append({
            'input_text': description,
            'target_smiles': smiles,
            'prompt_type': 'caption'
        })
        
        # Instruction format
        examples.append({
            'input_text': f"Generate SMILES for: {description}",
            'target_smiles': smiles,
            'prompt_type': 'instruction'
        })
        
        # Question format
        examples.append({
            'input_text': f"What is the SMILES representation of {description}?",
            'target_smiles': smiles,
            'prompt_type': 'question'
        })
        
        return examples
    
    def _initialize_curriculum(self):
        """Initialize curriculum learning ordering."""
        
        # Sort by difficulty for curriculum learning
        sorted_indices = np.argsort(self.difficulty_scores)
        
        # Create curriculum phases
        total_samples = len(self.data)
        easy_phase = int(total_samples * 0.3)    # 30% easiest
        medium_phase = int(total_samples * 0.6)  # 60% medium
        hard_phase = total_samples               # 100% all samples
        
        self.curriculum_phases = [
            sorted_indices[:easy_phase],        # Phase 1: Easy samples
            sorted_indices[:medium_phase],      # Phase 2: Easy + Medium
            sorted_indices[:hard_phase]         # Phase 3: All samples
        ]
        
        self.current_phase = 0
        self.active_indices = self.curriculum_phases[0]
        
        logger.info(f"Curriculum learning initialized:")
        logger.info(f"  Phase 1 (Easy): {len(self.curriculum_phases[0])} samples")
        logger.info(f"  Phase 2 (Medium): {len(self.curriculum_phases[1])} samples")
        logger.info(f"  Phase 3 (Hard): {len(self.curriculum_phases[2])} samples")
    
    def advance_curriculum_phase(self):
        """Advance to next curriculum phase."""
        if hasattr(self, 'curriculum_phases') and self.current_phase < len(self.curriculum_phases) - 1:
            self.current_phase += 1
            self.active_indices = self.curriculum_phases[self.current_phase]
            logger.info(f"Advanced to curriculum phase {self.current_phase + 1}")
            return True
        return False
    
    def __len__(self):
        if hasattr(self, 'active_indices'):
            return len(self.active_indices)
        return len(self.data)
    
    def __getitem__(self, idx):
        # Use curriculum learning indices if available
        if hasattr(self, 'active_indices'):
            actual_idx = self.active_indices[idx]
        else:
            actual_idx = idx
        
        item = self.data[actual_idx]
        
        # Tokenize input
        input_encoding = self.tokenizer(
            item['input_text'],
            max_length=self.config.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            item['target_smiles'],
            max_length=self.config.max_smiles_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze(),
            'target_attention_mask': target_encoding['attention_mask'].squeeze(),
            'raw_input': item['input_text'],
            'raw_target': item['target_smiles'],
            'prompt_type': item['prompt_type'],
            'difficulty': self.difficulty_scores[actual_idx] if hasattr(self, 'difficulty_scores') else 0.0
        }

class MolecularLoss(nn.Module):
    """Advanced loss function with molecular-specific objectives."""
    
    def __init__(self, tokenizer, config: TrainingConfig):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=0.1)
        
    def forward(self, logits, labels, batch_data=None):
        """Compute enhanced molecular loss."""
        
        # Base cross-entropy loss
        ce_loss = self.ce_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        total_loss = ce_loss
        loss_components = {'ce_loss': ce_loss.item()}
        
        if self.config.use_molecular_metrics and batch_data is not None:
            # Molecular validity bonus
            validity_bonus = self._compute_validity_bonus(logits, batch_data)
            total_loss = total_loss - 0.1 * validity_bonus
            loss_components['validity_bonus'] = validity_bonus.item()
        
        return total_loss, loss_components
    
    def _compute_validity_bonus(self, logits, batch_data):
        """Compute validity bonus for generated SMILES."""
        
        try:
            # Decode predictions
            pred_tokens = torch.argmax(logits, dim=-1)
            validity_bonus = 0.0
            valid_count = 0
            
            for i in range(pred_tokens.size(0)):
                pred_text = self.tokenizer.decode(pred_tokens[i], skip_special_tokens=True)
                
                # Check SMILES validity
                mol = Chem.MolFromSmiles(pred_text)
                if mol is not None:
                    validity_bonus += 1.0
                    valid_count += 1
            
            # Normalize by batch size
            if pred_tokens.size(0) > 0:
                validity_bonus = validity_bonus / pred_tokens.size(0)
            
            return torch.tensor(validity_bonus, device=logits.device)
            
        except Exception:
            return torch.tensor(0.0, device=logits.device)

class EnhancedTrainer:
    """Enhanced trainer with advanced training strategies."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize model and tokenizer
        logger.info(f"Loading model from {config.model_path}")
        self.tokenizer = T5Tokenizer.from_pretrained(config.model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(config.model_path)
        self.model.to(self.device)
        
        # Initialize loss function
        self.criterion = MolecularLoss(self.tokenizer, config)
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        self.training_stats = {
            'losses': [],
            'learning_rates': [],
            'molecular_metrics': [],
            'curriculum_phases': []
        }
        
        logger.info(f"Enhanced trainer initialized on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M")
    
    def _setup_optimizer_and_scheduler(self, train_dataloader):
        """Setup optimizer and learning rate scheduler."""
        
        # Optimizer with different learning rates for different components
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and 'encoder' in n],
                "weight_decay": self.config.weight_decay,
                "lr": self.config.learning_rate * 0.1  # Lower LR for encoder
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and 'decoder' in n],
                "weight_decay": self.config.weight_decay,
                "lr": self.config.learning_rate  # Standard LR for decoder
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.config.learning_rate
            },
        ]
        
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        total_steps = len(train_dataloader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"Training setup:")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Warmup steps: {warmup_steps}")
        logger.info(f"  Learning rate: {self.config.learning_rate}")
    
    def _evaluate_model(self, val_dataloader):
        """Evaluate model on validation set."""
        
        self.model.eval()
        total_loss = 0
        total_samples = 0
        molecular_metrics = {
            'valid_smiles': 0,
            'total_generated': 0,
            'avg_molecular_weight': 0,
            'avg_logp': 0
        }
        
        with torch.no_grad():
            for batch in val_dataloader:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Compute loss
                loss, _ = self.criterion(outputs.logits, labels, batch)
                total_loss += loss.item()
                total_samples += input_ids.size(0)
                
                # Generate samples for molecular metrics
                if molecular_metrics['total_generated'] < 100:  # Sample evaluation
                    generated_ids = self.model.generate(
                        input_ids=input_ids[:min(5, input_ids.size(0))],
                        attention_mask=attention_mask[:min(5, attention_mask.size(0))],
                        max_length=self.config.max_smiles_length,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                    
                    # Evaluate generated SMILES
                    for i in range(generated_ids.size(0)):
                        generated_smiles = self.tokenizer.decode(generated_ids[i], skip_special_tokens=True)
                        mol = Chem.MolFromSmiles(generated_smiles)
                        
                        molecular_metrics['total_generated'] += 1
                        
                        if mol is not None:
                            molecular_metrics['valid_smiles'] += 1
                            try:
                                molecular_metrics['avg_molecular_weight'] += Descriptors.MolWt(mol)
                                molecular_metrics['avg_logp'] += Descriptors.MolLogP(mol)
                            except Exception:
                                pass
        
        # Calculate final metrics
        avg_loss = total_loss / len(val_dataloader)
        
        if molecular_metrics['valid_smiles'] > 0:
            molecular_metrics['validity_rate'] = molecular_metrics['valid_smiles'] / molecular_metrics['total_generated']
            molecular_metrics['avg_molecular_weight'] /= molecular_metrics['valid_smiles']
            molecular_metrics['avg_logp'] /= molecular_metrics['valid_smiles']
        else:
            molecular_metrics['validity_rate'] = 0.0
        
        self.model.train()
        return avg_loss, molecular_metrics
    
    def train(self, train_dataset, val_dataset):
        """Main training loop with enhanced features."""
        
        # Create data loaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Setup optimizer and scheduler
        self._setup_optimizer_and_scheduler(train_dataloader)
        
        # Training loop
        logger.info("🚀 Starting enhanced training...")
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"\n--- EPOCH {epoch + 1}/{self.config.num_epochs} ---")
            
            # Curriculum learning phase advancement
            if (self.config.use_curriculum_learning and 
                hasattr(train_dataset, 'advance_curriculum_phase') and
                epoch > 0 and epoch % 3 == 0):  # Advance every 3 epochs
                
                if train_dataset.advance_curriculum_phase():
                    # Recreate dataloader with new curriculum phase
                    train_dataloader = DataLoader(
                        train_dataset,
                        batch_size=self.config.batch_size,
                        shuffle=True,
                        num_workers=2,
                        pin_memory=True
                    )
            
            # Training epoch
            epoch_loss = self._train_epoch(train_dataloader, epoch)
            
            # Evaluation
            if (epoch + 1) % 2 == 0:  # Evaluate every 2 epochs
                val_loss, molecular_metrics = self._evaluate_model(val_dataloader)
                
                logger.info(f"Validation Results:")
                logger.info(f"  Loss: {val_loss:.4f}")
                logger.info(f"  Validity Rate: {molecular_metrics['validity_rate']:.3f}")
                logger.info(f"  Avg MW: {molecular_metrics.get('avg_molecular_weight', 0):.1f}")
                logger.info(f"  Avg LogP: {molecular_metrics.get('avg_logp', 0):.2f}")
                
                # Save best model
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self._save_model('best_model')
                    logger.info(f"🎉 New best model saved! Loss: {val_loss:.4f}")
                
                # Store metrics
                self.training_stats['molecular_metrics'].append(molecular_metrics)
        
        # Final model save
        self._save_model('final_model')
        self._save_training_stats()
        
        logger.info("🎉 Enhanced training completed!")
        return self.training_stats
    
    def _train_epoch(self, dataloader, epoch):
        """Train for one epoch."""
        
        self.model.train()
        total_loss = 0
        accumulated_loss = 0
        
        for step, batch in enumerate(dataloader):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Compute loss
            loss, loss_components = self.criterion(outputs.logits, labels, batch)
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            accumulated_loss += loss.item()
            
            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Logging
                self.global_step += 1
                total_loss += accumulated_loss
                
                if self.global_step % self.config.logging_steps == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    logger.info(f"Step {self.global_step}: Loss {accumulated_loss:.4f}, LR {current_lr:.2e}")
                    
                    # Store training stats
                    self.training_stats['losses'].append(accumulated_loss)
                    self.training_stats['learning_rates'].append(current_lr)
                
                accumulated_loss = 0
        
        avg_loss = total_loss / (len(dataloader) // self.config.gradient_accumulation_steps)
        logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def _save_model(self, name: str):
        """Save model checkpoint."""
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = output_dir / name
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        # Save training config
        with open(model_path / 'training_config.json', 'w') as f:
            config_dict = {
                'global_step': self.global_step,
                'best_loss': self.best_loss,
                'config': self.config.__dict__
            }
            json.dump(config_dict, f, indent=2)
    
    def _save_training_stats(self):
        """Save training statistics."""
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'training_stats.json', 'w') as f:
            json.dump(self.training_stats, f, indent=2, default=str)

def run_enhanced_training():
    """Run the complete enhanced training pipeline."""
    
    logger.info("🧬 ENHANCED MOLECULAR TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # Load configuration
    config = TrainingConfig()
    
    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(config.model_path)
    
    # Create datasets
    logger.info("Creating enhanced datasets...")
    train_dataset = MolecularDataset(config.train_data, tokenizer, config, is_training=True)
    val_dataset = MolecularDataset(config.val_data, tokenizer, config, is_training=False)
    
    # Create trainer
    trainer = EnhancedTrainer(config)
    
    # Start training
    training_stats = trainer.train(train_dataset, val_dataset)
    
    logger.info("🎉 Enhanced training pipeline completed!")
    return training_stats

def main():
    """Main function."""
    
    try:
        # Run enhanced training
        stats = run_enhanced_training()
        
        logger.info("Training completed successfully!")
        logger.info(f"Final statistics:")
        logger.info(f"  Total training steps: {len(stats['losses'])}")
        logger.info(f"  Best validation loss: {min(stats['losses']):.4f}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    stats = main()