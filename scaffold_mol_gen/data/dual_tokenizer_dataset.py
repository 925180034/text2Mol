"""
修改后的数据集类，支持文本和SMILES使用不同的tokenizer
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

class DualTokenizerScaffoldMolDataset(Dataset):
    """支持双tokenizer的数据集"""
    
    def __init__(self, 
                 data_path: str,
                 text_tokenizer: Any,  # SciBERT tokenizer
                 smiles_tokenizer: Any,  # T5 tokenizer
                 input_modalities: List[str] = ['text', 'smiles'],
                 output_modality: str = 'smiles',
                 max_text_length: int = 256,
                 max_smiles_length: int = 128,
                 image_size: Tuple[int, int] = (224, 224),
                 scaffold_type: str = 'murcko',
                 augment_data: bool = False,
                 filter_invalid: bool = True):
        
        self.data_path = Path(data_path)
        self.text_tokenizer = text_tokenizer
        self.smiles_tokenizer = smiles_tokenizer
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
        logger.info(f"Text tokenizer: {type(text_tokenizer).__name__}")
        logger.info(f"SMILES tokenizer: {type(smiles_tokenizer).__name__}")
    
    def _load_data(self) -> pd.DataFrame:
        """Load and preprocess data from CSV file."""
        try:
            # Load CSV
            df = pd.read_csv(self.data_path)
            
            # Required columns
            required_cols = []
            if 'text' in self.input_modalities:
                # Support both 'text' and 'description' columns
                if 'text' in df.columns:
                    required_cols.append('text')
                elif 'description' in df.columns:
                    required_cols.append('description')
                    # Rename for internal consistency
                    df['text'] = df['description']
                else:
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
                # 使用SciBERT tokenizer处理文本
                text_tokens = self.text_tokenizer(
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
                # 使用T5 tokenizer处理SMILES
                smiles_tokens = self.smiles_tokenizer(
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
                # 使用T5 tokenizer处理scaffold SMILES
                scaffold_tokens = self.smiles_tokenizer(
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
                # 使用T5 tokenizer处理目标SMILES
                target_tokens = self.smiles_tokenizer(
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
    
    def get_vocab_size(self) -> Dict[str, int]:
        """Get tokenizer vocabulary sizes."""
        return {
            'text': len(self.text_tokenizer),
            'smiles': len(self.smiles_tokenizer)
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            'total_samples': len(self.data),
            'input_modalities': self.input_modalities,
            'output_modality': self.output_modality,
            'text_tokenizer_vocab': len(self.text_tokenizer),
            'smiles_tokenizer_vocab': len(self.smiles_tokenizer)
        }
        
        if 'SMILES' in self.data.columns:
            # Molecular property statistics
            properties = []
            for smiles in self.data['SMILES'].tolist()[:1000]:  # Sample for efficiency
                props = MolecularUtils.compute_molecular_properties(smiles)
                if props:
                    properties.append(props)
            
            if properties:
                import pandas as pd
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
