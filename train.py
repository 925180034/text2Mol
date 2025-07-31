#!/usr/bin/env python3
"""
Training script for scaffold-based molecular generation.

This script handles the complete training pipeline including:
- Data loading and preprocessing
- Model initialization
- Training orchestration
- Evaluation and checkpointing
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import yaml
import torch
from transformers import T5Tokenizer

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from scaffold_mol_gen.models.core_model import ScaffoldBasedMolT5Generator
from scaffold_mol_gen.data.dataset import ScaffoldMolDataset
from scaffold_mol_gen.data.preprocessing import MolecularPreprocessor
from scaffold_mol_gen.training.trainer import create_trainer
from scaffold_mol_gen.utils.mol_utils import MolecularUtils

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train scaffold-based molecular generation model')
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/training_config.yaml',
        help='Path to training configuration file'
    )
    
    parser.add_argument(
        '--train-data',
        type=str,
        help='Path to training data CSV file (overrides config)'
    )
    
    parser.add_argument(
        '--val-data',
        type=str,
        help='Path to validation data CSV file (overrides config)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Output directory for checkpoints and logs'
    )
    
    parser.add_argument(
        '--resume-from-checkpoint',
        type=str,
        help='Path to checkpoint to resume training from'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with reduced dataset size'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform dry run without actual training'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file with base config support."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Handle base config inheritance
        if 'base_config' in config:
            base_config_path = os.path.join(os.path.dirname(config_path), config['base_config'])
            if os.path.exists(base_config_path):
                with open(base_config_path, 'r') as f:
                    base_config = yaml.safe_load(f)
                
                # Merge configs (current config overrides base config)
                merged_config = deep_merge_dict(base_config, config)
                config = merged_config
                logger.info(f"Merged base config from {base_config_path}")
            else:
                logger.warning(f"Base config file not found: {base_config_path}")
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise

def deep_merge_dict(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key == 'base_config':
            continue  # Skip base_config key
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = value
    return result

def setup_reproducibility(seed: int):
    """Setup reproducibility settings."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    
    logger.info(f"Set random seed to {seed}")

def create_datasets(config: dict, tokenizer, debug: bool = False):
    """Create training and validation datasets."""
    data_config = config['data']
    
    # Determine dataset paths
    train_path = config.get('train_data') or data_config['train_data']
    val_path = config.get('val_data') or data_config['val_data']
    
    logger.info(f"Loading training data from: {train_path}")
    logger.info(f"Loading validation data from: {val_path}")
    
    # Dataset parameters
    dataset_params = {
        'tokenizer': tokenizer,
        'input_modalities': data_config['input_modalities'],
        'output_modality': data_config['output_modality'],
        'max_text_length': data_config['max_text_length'],
        'max_smiles_length': data_config['max_smiles_length'],
        'image_size': tuple(data_config['image_size']),
        'scaffold_type': data_config['scaffold_type'],
        'filter_invalid': data_config['filter_invalid']
    }
    
    # Create datasets
    train_dataset = ScaffoldMolDataset(
        data_path=train_path,
        **dataset_params
    )
    
    val_dataset = ScaffoldMolDataset(
        data_path=val_path,
        **dataset_params
    )
    
    # Debug mode: use smaller datasets
    if debug:
        logger.info("Debug mode: Using smaller datasets")
        # Limit dataset size for debugging
        train_dataset.data = train_dataset.data.head(100)
        val_dataset.data = val_dataset.data.head(50)
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    return train_dataset, val_dataset

def create_model(config: dict) -> ScaffoldBasedMolT5Generator:
    """Create and initialize the model."""
    model_config = config['model']
    
    logger.info("Creating ScaffoldBasedMolT5Generator model...")
    model = ScaffoldBasedMolT5Generator(model_config)
    
    # Print model information
    model_info = model.get_model_info()
    logger.info(f"Model: {model_info['model_name']} v{model_info['version']}")
    logger.info(f"Total parameters: {model_info['total_parameters']:,}")
    logger.info(f"Trainable parameters: {model_info['trainable_parameters']:,}")
    
    return model

def preprocess_data(config: dict):
    """Preprocess training data if needed."""
    data_config = config['data']
    
    if data_config.get('preprocess', False):
        logger.info("Preprocessing training data...")
        
        preprocessor = MolecularPreprocessor(
            canonicalize=True,
            remove_salts=True,
            filter_invalid=True,
            remove_duplicates=True
        )
        
        # Preprocess training data
        train_path = data_config['train_data']
        if os.path.exists(train_path):
            import pandas as pd
            df = pd.read_csv(train_path)
            processed_df = preprocessor.preprocess_dataframe(df)
            
            # Save preprocessed data
            processed_path = train_path.replace('.csv', '_preprocessed.csv')
            processed_df.to_csv(processed_path, index=False)
            
            # Update config
            config['data']['train_data'] = processed_path
            
            logger.info(f"Preprocessed data saved to: {processed_path}")
            logger.info(f"Preprocessing stats: {preprocessor.get_preprocessing_stats()}")

def main():
    """Main training function."""
    args = parse_args()
    
    # Setup reproducibility
    setup_reproducibility(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.train_data:
        config['train_data'] = args.train_data
    if args.val_data:
        config['val_data'] = args.val_data
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config['output_dir'] = str(output_dir)
    
    logger.info("="*50)
    logger.info("SCAFFOLD-BASED MOLECULAR GENERATION TRAINING")
    logger.info("="*50)
    
    # Preprocess data if needed
    preprocess_data(config)
    
    # Initialize tokenizer
    molt5_checkpoint = config['model']['molt5_checkpoint']
    logger.info(f"Loading tokenizer from: {molt5_checkpoint}")
    tokenizer = T5Tokenizer.from_pretrained(molt5_checkpoint)
    
    # Create datasets
    train_dataset, val_dataset = create_datasets(config, tokenizer, args.debug)
    
    # Print dataset statistics
    train_stats = train_dataset.get_statistics()
    logger.info(f"Training dataset statistics: {train_stats}")
    
    # Create model
    model = create_model(config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if args.dry_run:
        logger.info("Dry run completed successfully!")
        return
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config['training'],
        output_dir=str(output_dir),
        device=device,
        logger_type=config.get('logging', {}).get('logger_type', 'tensorboard')
    )
    
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        logger.info(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
        trainer.load_checkpoint(args.resume_from_checkpoint)
    
    # Start training
    logger.info("Starting training...")
    try:
        results = trainer.train()
        
        # Log final results
        logger.info("Training completed successfully!")
        logger.info(f"Training time: {results['training_time']:.2f}s")
        logger.info(f"Best validation loss: {results['best_validation_loss']:.4f}")
        
        # Save final configuration
        config_path = output_dir / 'final_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Final configuration saved to: {config_path}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    logger.info("Training script completed!")

if __name__ == '__main__':
    main()