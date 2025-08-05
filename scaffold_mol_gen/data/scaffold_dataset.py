"""
Simple dataset class for scaffold-based molecular generation training.
"""

import logging
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, Any
from pathlib import Path
from transformers import T5Tokenizer, BertTokenizer
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

logger = logging.getLogger(__name__)

class ScaffoldDataset(Dataset):
    """
    Simple dataset for scaffold-based molecular generation.
    Handles loading and preprocessing of molecular data with scaffolds.
    """
    
    def __init__(self, 
                 data_path: str,
                 max_text_length: int = 256,
                 max_smiles_length: int = 128,
                 cache_data: bool = True,
                 molt5_path: str = "/root/autodl-tmp/text2Mol-models/MolT5-Large-Caption2SMILES",
                 bert_path: str = "/root/autodl-tmp/text2Mol-models/bert-base-uncased"):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to CSV file
            max_text_length: Maximum text sequence length
            max_smiles_length: Maximum SMILES sequence length
            cache_data: Whether to cache processed data
        """
        self.data_path = Path(data_path)
        self.max_text_length = max_text_length
        self.max_smiles_length = max_smiles_length
        self.cache_data = cache_data
        
        # Initialize tokenizers
        logger.info("Initializing tokenizers...")
        self.smiles_tokenizer = T5Tokenizer.from_pretrained(molt5_path)
        self.text_tokenizer = BertTokenizer.from_pretrained(bert_path)
        
        # Load data
        logger.info(f"Loading data from {data_path}...")
        self.df = pd.read_csv(data_path)
        
        # Filter out invalid entries
        initial_size = len(self.df)
        self.df = self.df.dropna(subset=['SMILES'])
        if 'description' in self.df.columns:
            self.df = self.df.dropna(subset=['description'])
        elif 'text' in self.df.columns:
            self.df = self.df.dropna(subset=['text'])
            self.df['description'] = self.df['text']
        
        # Validate SMILES
        valid_indices = []
        for idx, row in self.df.iterrows():
            try:
                mol = Chem.MolFromSmiles(row['SMILES'])
                if mol is not None:
                    valid_indices.append(idx)
            except:
                continue
        
        self.df = self.df.loc[valid_indices].reset_index(drop=True)
        logger.info(f"Filtered {initial_size} -> {len(self.df)} valid samples")
        
        # Extract scaffolds
        logger.info("Extracting scaffolds...")
        self.df['scaffold'] = self.df['SMILES'].apply(self._extract_scaffold)
        
        # Cache processed data if requested
        self.cache = {} if cache_data else None
        
    def _extract_scaffold(self, smiles: str) -> str:
        """Extract Murcko scaffold from SMILES."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return smiles  # Return original if can't parse
            
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold)
            return scaffold_smiles
        except:
            return smiles  # Return original on error
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        # Check cache
        if self.cache is not None and idx in self.cache:
            return self.cache[idx]
        
        row = self.df.iloc[idx]
        
        # Get text and SMILES
        text = row.get('description', row.get('text', ''))
        target_smiles = row['SMILES']
        scaffold_smiles = row['scaffold']
        
        # Tokenize text
        text_inputs = self.text_tokenizer(
            text,
            max_length=self.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize scaffold SMILES (input)
        scaffold_inputs = self.smiles_tokenizer(
            scaffold_smiles,
            max_length=self.max_smiles_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target SMILES (output)
        target_inputs = self.smiles_tokenizer(
            target_smiles,
            max_length=self.max_smiles_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Prepare sample
        sample = {
            # Text inputs
            'text_input_ids': text_inputs['input_ids'].squeeze(0),
            'text_attention_mask': text_inputs['attention_mask'].squeeze(0),
            
            # Scaffold inputs
            'scaffold_input_ids': scaffold_inputs['input_ids'].squeeze(0),
            'scaffold_attention_mask': scaffold_inputs['attention_mask'].squeeze(0),
            
            # Target outputs
            'labels': target_inputs['input_ids'].squeeze(0),
            'target_attention_mask': target_inputs['attention_mask'].squeeze(0),
            
            # Metadata
            'text': text,
            'scaffold_smiles': scaffold_smiles,
            'target_smiles': target_smiles,
        }
        
        # Cache if enabled
        if self.cache is not None:
            self.cache[idx] = sample
        
        return sample
    
    def collate_fn(self, batch):
        """Custom collate function for batching."""
        # Stack tensors
        text_input_ids = torch.stack([x['text_input_ids'] for x in batch])
        text_attention_mask = torch.stack([x['text_attention_mask'] for x in batch])
        scaffold_input_ids = torch.stack([x['scaffold_input_ids'] for x in batch])
        scaffold_attention_mask = torch.stack([x['scaffold_attention_mask'] for x in batch])
        labels = torch.stack([x['labels'] for x in batch])
        target_attention_mask = torch.stack([x['target_attention_mask'] for x in batch])
        
        return {
            'text_input_ids': text_input_ids,
            'text_attention_mask': text_attention_mask,
            'scaffold_input_ids': scaffold_input_ids,
            'scaffold_attention_mask': scaffold_attention_mask,
            'labels': labels,
            'target_attention_mask': target_attention_mask,
            'texts': [x['text'] for x in batch],
            'scaffold_smiles': [x['scaffold_smiles'] for x in batch],
            'target_smiles': [x['target_smiles'] for x in batch],
        }