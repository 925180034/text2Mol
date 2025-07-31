"""
Dataset classes for scaffold-based molecular generation.

This module provides PyTorch Dataset classes for handling multi-modal
molecular data with scaffold preservation.
"""

import logging
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
from pathlib import Path

from ..utils.scaffold_utils import ScaffoldExtractor
from ..utils.mol_utils import MolecularUtils, smiles_to_graph, smiles_to_image

logger = logging.getLogger(__name__)

class ScaffoldMolDataset(Dataset):
    """
    Dataset for scaffold-based molecular generation.
    
    Supports multiple input-output combinations:
    1. Text → SMILES
    2. SMILES → SMILES  
    3. Graph → SMILES
    4. Image → SMILES
    5. Text + SMILES → SMILES
    6. Text + Graph → SMILES
    7. Text + Image → SMILES
    """
    
    def __init__(self, 
                 data_path: str,
                 tokenizer: Any,
                 input_modalities: List[str] = ['text', 'smiles'],
                 output_modality: str = 'smiles',
                 max_text_length: int = 256,
                 max_smiles_length: int = 128,
                 image_size: Tuple[int, int] = (224, 224),
                 scaffold_type: str = 'murcko',
                 augment_data: bool = False,
                 filter_invalid: bool = True):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to CSV file containing the data
            tokenizer: Tokenizer for text/SMILES processing
            input_modalities: List of input modalities to use
            output_modality: Target output modality
            max_text_length: Maximum text sequence length
            max_smiles_length: Maximum SMILES sequence length
            image_size: Image dimensions for molecular images
            scaffold_type: Type of scaffold extraction
            augment_data: Whether to apply data augmentation
            filter_invalid: Whether to filter invalid molecules
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.input_modalities = input_modalities
        self.output_modality = output_modality
        self.max_text_length = max_text_length
        self.max_smiles_length = max_smiles_length
        self.image_size = image_size
        self.scaffold_type = scaffold_type
        self.augment_data = augment_data
        self.filter_invalid = filter_invalid
        
        # Initialize scaffold extractor
        self.scaffold_extractor = ScaffoldExtractor()
        
        # Load and preprocess data
        self.data = self._load_data()
        
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
        logger.info(f"Input modalities: {input_modalities}")
        logger.info(f"Output modality: {output_modality}")
    
    def _load_data(self) -> pd.DataFrame:
        """Load and preprocess data from CSV file."""
        try:
            # Load CSV
            df = pd.read_csv(self.data_path)
            
            # Required columns
            required_cols = []
            if 'text' in self.input_modalities:
                required_cols.append('text')
            if 'smiles' in self.input_modalities or self.output_modality == 'smiles':
                required_cols.append('SMILES')
            
            # Check for required columns
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Filter invalid molecules if requested
            if self.filter_invalid and 'SMILES' in df.columns:
                valid_mask = df['SMILES'].apply(MolecularUtils.validate_smiles)
                df = df[valid_mask].reset_index(drop=True)
                logger.info(f"Filtered to {len(df)} valid molecules")
            
            # Extract scaffolds
            if 'SMILES' in df.columns:
                scaffolds = self.scaffold_extractor.batch_extract_scaffolds(
                    df['SMILES'].tolist(), 
                    scaffold_type=self.scaffold_type
                )
                df['scaffold'] = scaffolds
                
                # Filter molecules with valid scaffolds
                valid_scaffold_mask = df['scaffold'].notna()
                df = df[valid_scaffold_mask].reset_index(drop=True)
                logger.info(f"Found {len(df)} molecules with valid scaffolds")
            
            # Canonicalize SMILES
            if 'SMILES' in df.columns:
                df['SMILES'] = df['SMILES'].apply(
                    lambda x: MolecularUtils.canonicalize_smiles(x) or x
                )
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample from the dataset."""
        row = self.data.iloc[idx]
        sample = {'index': idx}
        
        try:
            # Process input modalities
            if 'text' in self.input_modalities and 'text' in row:
                text_tokens = self.tokenizer(
                    str(row['text']),
                    max_length=self.max_text_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                sample['text_tokens'] = {
                    'input_ids': text_tokens['input_ids'].squeeze(0),
                    'attention_mask': text_tokens['attention_mask'].squeeze(0)
                }
            
            if 'smiles' in self.input_modalities and 'SMILES' in row:
                smiles_tokens = self.tokenizer(
                    str(row['SMILES']),
                    max_length=self.max_smiles_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                sample['smiles_tokens'] = {
                    'input_ids': smiles_tokens['input_ids'].squeeze(0),
                    'attention_mask': smiles_tokens['attention_mask'].squeeze(0)
                }
            
            # Process scaffold
            if 'scaffold' in row and row['scaffold']:
                scaffold_tokens = self.tokenizer(
                    str(row['scaffold']),
                    max_length=self.max_smiles_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                sample['scaffold_tokens'] = {
                    'input_ids': scaffold_tokens['input_ids'].squeeze(0),
                    'attention_mask': scaffold_tokens['attention_mask'].squeeze(0)
                }
                
                # Convert scaffold to graph if needed
                if 'graph' in self.input_modalities:
                    scaffold_graph = smiles_to_graph(str(row['scaffold']))
                    if scaffold_graph:
                        sample['scaffold_graph'] = scaffold_graph
                
                # Convert scaffold to image if needed  
                if 'image' in self.input_modalities:
                    scaffold_image = smiles_to_image(str(row['scaffold']), self.image_size)
                    if scaffold_image is not None:
                        sample['scaffold_image'] = scaffold_image
            
            # Process target output
            if self.output_modality == 'smiles' and 'SMILES' in row:
                target_tokens = self.tokenizer(
                    str(row['SMILES']),
                    max_length=self.max_smiles_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                sample['target_tokens'] = {
                    'input_ids': target_tokens['input_ids'].squeeze(0),
                    'attention_mask': target_tokens['attention_mask'].squeeze(0)
                }
                
                # Also include target graph and image for multi-task learning
                target_graph = smiles_to_graph(str(row['SMILES']))
                if target_graph:
                    sample['target_graph'] = target_graph
                
                target_image = smiles_to_image(str(row['SMILES']), self.image_size)
                if target_image is not None:
                    sample['target_image'] = target_image
            
            # Include raw strings for evaluation
            sample['raw_data'] = {
                'text': str(row.get('text', '')),
                'smiles': str(row.get('SMILES', '')),
                'scaffold': str(row.get('scaffold', ''))
            }
            
            return sample
            
        except Exception as e:
            logger.warning(f"Error processing sample {idx}: {e}")
            # Return empty sample to avoid breaking training
            return {'index': idx, 'raw_data': {'text': '', 'smiles': '', 'scaffold': ''}}
    
    def get_vocab_size(self) -> int:
        """Get tokenizer vocabulary size."""
        return len(self.tokenizer)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            'total_samples': len(self.data),
            'input_modalities': self.input_modalities,
            'output_modality': self.output_modality
        }
        
        if 'SMILES' in self.data.columns:
            # Molecular property statistics
            properties = []
            for smiles in self.data['SMILES'].tolist()[:1000]:  # Sample for efficiency
                props = MolecularUtils.compute_molecular_properties(smiles)
                if props:
                    properties.append(props)
            
            if properties:
                prop_df = pd.DataFrame(properties)
                stats['molecular_properties'] = {
                    'mean_molecular_weight': float(prop_df['molecular_weight'].mean()),
                    'mean_logp': float(prop_df['logp'].mean()),
                    'mean_tpsa': float(prop_df['tpsa'].mean())
                }
        
        if 'scaffold' in self.data.columns:
            # Scaffold statistics
            scaffold_stats = self.scaffold_extractor.scaffold_statistics(
                self.data['scaffold'].dropna().tolist()
            )
            stats['scaffold_statistics'] = scaffold_stats
        
        return stats


class MultiModalMolDataset(Dataset):
    """
    Multi-modal molecular dataset supporting all seven input-output combinations
    specified in the requirements.
    """
    
    def __init__(self,
                 data_path: str,
                 tokenizer: Any,
                 combination_type: int = 1,
                 **kwargs):
        """
        Initialize multi-modal dataset.
        
        Args:
            data_path: Path to data file
            tokenizer: Tokenizer for text/SMILES
            combination_type: Input-output combination (1-7)
            **kwargs: Additional arguments for base dataset
        """
        self.combination_type = combination_type
        
        # Define input-output combinations
        self.combinations = {
            1: (['text'], 'smiles'),           # Text → SMILES
            2: (['smiles'], 'smiles'),         # SMILES → SMILES
            3: (['graph'], 'smiles'),          # Graph → SMILES 
            4: (['image'], 'smiles'),          # Image → SMILES
            5: (['text', 'smiles'], 'smiles'), # Text + SMILES → SMILES
            6: (['text', 'graph'], 'smiles'),  # Text + Graph → SMILES
            7: (['text', 'image'], 'smiles')   # Text + Image → SMILES
        }
        
        if combination_type not in self.combinations:
            raise ValueError(f"Invalid combination type: {combination_type}")
        
        input_modalities, output_modality = self.combinations[combination_type]
        
        # Initialize base dataset
        super().__init__(
            data_path=data_path,
            tokenizer=tokenizer,
            input_modalities=input_modalities,
            output_modality=output_modality,
            **kwargs
        )
        
        logger.info(f"Initialized combination {combination_type}: {input_modalities} → {output_modality}")


class ConversationalMolDataset(Dataset):
    """
    Dataset for multi-turn conversational molecular design.
    Supports maintaining context across multiple generations.
    """
    
    def __init__(self,
                 data_path: str,
                 tokenizer: Any,
                 max_turns: int = 5,
                 context_length: int = 512,
                 **kwargs):
        """
        Initialize conversational dataset.
        
        Args:
            data_path: Path to conversation data
            tokenizer: Tokenizer for processing
            max_turns: Maximum conversation turns
            context_length: Maximum context sequence length
            **kwargs: Additional arguments
        """
        self.max_turns = max_turns
        self.context_length = context_length
        
        # Load conversation data
        self.conversations = self._load_conversations(data_path)
        
        super().__init__(
            data_path=data_path,
            tokenizer=tokenizer,
            **kwargs
        )
    
    def _load_conversations(self, data_path: str) -> List[Dict]:
        """Load conversation data."""
        # Implementation for loading multi-turn conversation data
        # This would depend on the specific format of conversation data
        return []
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get conversation sample with context."""
        # Implementation for conversation-aware sampling
        sample = super().__getitem__(idx)
        
        # Add conversation context
        # This would include previous turns in the conversation
        
        return sample


class BalancedScaffoldDataset(Dataset):
    """
    Dataset with balanced sampling across different scaffold types
    to ensure diverse training.
    """
    
    def __init__(self,
                 data_path: str,
                 tokenizer: Any,
                 balance_strategy: str = 'scaffold',
                 min_samples_per_scaffold: int = 5,
                 **kwargs):
        """
        Initialize balanced dataset.
        
        Args:
            data_path: Path to data
            tokenizer: Tokenizer
            balance_strategy: Balancing strategy ('scaffold', 'property')
            min_samples_per_scaffold: Minimum samples per scaffold group
            **kwargs: Additional arguments
        """
        self.balance_strategy = balance_strategy
        self.min_samples_per_scaffold = min_samples_per_scaffold
        
        # Initialize base dataset
        base_dataset = ScaffoldMolDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            **kwargs
        )
        
        # Create balanced sampling indices
        self.indices = self._create_balanced_indices(base_dataset)
        self.base_dataset = base_dataset
        
        logger.info(f"Created balanced dataset with {len(self.indices)} samples")
    
    def _create_balanced_indices(self, base_dataset: ScaffoldMolDataset) -> List[int]:
        """Create balanced sampling indices."""
        if self.balance_strategy == 'scaffold':
            # Group by scaffold
            scaffold_groups = {}
            for idx, row in base_dataset.data.iterrows():
                scaffold = row.get('scaffold', '')
                if scaffold not in scaffold_groups:
                    scaffold_groups[scaffold] = []
                scaffold_groups[scaffold].append(idx)
            
            # Sample equally from each scaffold group
            balanced_indices = []
            for scaffold, indices in scaffold_groups.items():
                if len(indices) >= self.min_samples_per_scaffold:
                    # Include all samples for scaffolds with enough data
                    balanced_indices.extend(indices)
                elif len(indices) > 0:
                    # Oversample scaffolds with few samples
                    repeats = self.min_samples_per_scaffold // len(indices) + 1
                    extended_indices = indices * repeats
                    balanced_indices.extend(extended_indices[:self.min_samples_per_scaffold])
            
            return balanced_indices
        
        else:
            # Return all indices for other strategies
            return list(range(len(base_dataset)))
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        actual_idx = self.indices[idx]
        return self.base_dataset[actual_idx]