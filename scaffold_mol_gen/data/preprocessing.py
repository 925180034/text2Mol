"""
Data preprocessing and augmentation utilities.

This module provides tools for preprocessing molecular data,
including augmentation, normalization, and validation.
"""

import logging
import random
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Any, Tuple, Union
from rdkit import Chem
from rdkit.Chem import AllChem

from ..utils.mol_utils import MolecularUtils, smiles_to_graph, smiles_to_image
from ..utils.scaffold_utils import ScaffoldExtractor

logger = logging.getLogger(__name__)

class MolecularPreprocessor:
    """Comprehensive molecular data preprocessing pipeline."""
    
    def __init__(self, 
                 canonicalize: bool = True,
                 remove_salts: bool = True,
                 standardize: bool = True,
                 filter_invalid: bool = True,
                 remove_duplicates: bool = True,
                 min_atoms: int = 3,
                 max_atoms: int = 100):
        """
        Initialize preprocessor.
        
        Args:
            canonicalize: Whether to canonicalize SMILES
            remove_salts: Whether to remove salt components
            standardize: Whether to standardize molecules
            filter_invalid: Whether to filter invalid molecules
            remove_duplicates: Whether to remove duplicate SMILES
            min_atoms: Minimum number of atoms
            max_atoms: Maximum number of atoms
        """
        self.canonicalize = canonicalize
        self.remove_salts = remove_salts
        self.standardize = standardize
        self.filter_invalid = filter_invalid
        self.remove_duplicates = remove_duplicates
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms
        
        # Statistics
        self.stats = {
            'original_count': 0,
            'invalid_removed': 0,
            'salts_removed': 0,
            'duplicates_removed': 0,
            'size_filtered': 0,
            'final_count': 0
        }
    
    def preprocess_dataframe(self, df: pd.DataFrame, 
                           smiles_column: str = 'SMILES') -> pd.DataFrame:
        """
        Preprocess a pandas DataFrame containing SMILES.
        
        Args:
            df: Input DataFrame
            smiles_column: Name of SMILES column
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info(f"Starting preprocessing of {len(df)} molecules")
        self.stats['original_count'] = len(df)
        
        # Make a copy to avoid modifying original
        processed_df = df.copy()
        
        # Step 1: Filter invalid SMILES
        if self.filter_invalid:
            valid_mask = processed_df[smiles_column].apply(self._is_valid_smiles)
            invalid_count = (~valid_mask).sum()
            processed_df = processed_df[valid_mask].reset_index(drop=True)
            self.stats['invalid_removed'] = invalid_count
            logger.info(f"Removed {invalid_count} invalid SMILES")
        
        # Step 2: Remove salts
        if self.remove_salts:
            original_count = len(processed_df)
            processed_df[smiles_column] = processed_df[smiles_column].apply(
                self._remove_salts
            )
            # Filter out None values (molecules that couldn't be desalted)
            processed_df = processed_df[processed_df[smiles_column].notna()].reset_index(drop=True)
            salts_removed = original_count - len(processed_df)
            self.stats['salts_removed'] = salts_removed
            logger.info(f"Processed salt removal for {original_count} molecules, removed {salts_removed}")
        
        # Step 3: Canonicalize SMILES
        if self.canonicalize:
            processed_df[smiles_column] = processed_df[smiles_column].apply(
                self._canonicalize_smiles
            )
            # Filter out None values
            processed_df = processed_df[processed_df[smiles_column].notna()].reset_index(drop=True)
        
        # Step 4: Filter by molecular size
        original_count = len(processed_df)
        size_mask = processed_df[smiles_column].apply(
            lambda x: self._check_molecular_size(x)
        )
        processed_df = processed_df[size_mask].reset_index(drop=True)
        size_filtered = original_count - len(processed_df)
        self.stats['size_filtered'] = size_filtered
        logger.info(f"Filtered {size_filtered} molecules by size")
        
        # Step 5: Remove duplicates
        if self.remove_duplicates:
            original_count = len(processed_df)
            processed_df = processed_df.drop_duplicates(subset=[smiles_column]).reset_index(drop=True)
            duplicates_removed = original_count - len(processed_df)
            self.stats['duplicates_removed'] = duplicates_removed
            logger.info(f"Removed {duplicates_removed} duplicate SMILES")
        
        self.stats['final_count'] = len(processed_df)
        
        logger.info(f"Preprocessing complete: {self.stats['original_count']} → {self.stats['final_count']} molecules")
        
        return processed_df
    
    def _is_valid_smiles(self, smiles: str) -> bool:
        """Check if SMILES is valid."""
        try:
            return MolecularUtils.validate_smiles(str(smiles))
        except:
            return False
    
    def _remove_salts(self, smiles: str) -> Optional[str]:
        """Remove salt components from SMILES."""
        try:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                return None
            
            # Remove salt components (keep largest fragment)
            from rdkit.Chem.SaltRemover import SaltRemover
            remover = SaltRemover()
            mol = remover.StripMol(mol)
            
            if mol is None or mol.GetNumAtoms() == 0:
                return None
            
            return Chem.MolToSmiles(mol)
            
        except Exception as e:
            logger.warning(f"Error removing salts from {smiles}: {e}")
            return None
    
    def _canonicalize_smiles(self, smiles: str) -> Optional[str]:
        """Canonicalize SMILES string."""
        try:
            return MolecularUtils.canonicalize_smiles(str(smiles))
        except:
            return None
    
    def _check_molecular_size(self, smiles: str) -> bool:
        """Check if molecule is within size limits."""
        try:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                return False
            
            num_atoms = mol.GetNumAtoms()
            return self.min_atoms <= num_atoms <= self.max_atoms
            
        except:
            return False
    
    def get_preprocessing_stats(self) -> Dict[str, int]:
        """Get preprocessing statistics."""
        return self.stats.copy()


class DataAugmentation:
    """Data augmentation techniques for molecular data."""
    
    def __init__(self, 
                 smiles_enumeration: bool = True,
                 scaffold_hopping: bool = False,
                 text_augmentation: bool = True,
                 augmentation_factor: int = 2):
        """
        Initialize data augmentation.
        
        Args:
            smiles_enumeration: Whether to enumerate SMILES representations
            scaffold_hopping: Whether to perform scaffold hopping
            text_augmentation: Whether to augment text descriptions
            augmentation_factor: Factor by which to increase data size
        """
        self.smiles_enumeration = smiles_enumeration
        self.scaffold_hopping = scaffold_hopping
        self.text_augmentation = text_augmentation
        self.augmentation_factor = augmentation_factor
    
    def augment_dataset(self, df: pd.DataFrame,
                       smiles_column: str = 'SMILES',
                       text_column: str = 'text') -> pd.DataFrame:
        """
        Augment dataset with various techniques.
        
        Args:
            df: Input DataFrame
            smiles_column: Name of SMILES column
            text_column: Name of text column
            
        Returns:
            Augmented DataFrame
        """
        augmented_data = []
        
        for _, row in df.iterrows():
            # Keep original
            augmented_data.append(row.to_dict())
            
            # Generate augmented versions
            for i in range(self.augmentation_factor - 1):
                augmented_row = row.to_dict()
                
                # SMILES enumeration
                if self.smiles_enumeration and smiles_column in row:
                    enumerated_smiles = self._enumerate_smiles(row[smiles_column])
                    if enumerated_smiles:
                        augmented_row[smiles_column] = enumerated_smiles
                
                # Text augmentation
                if self.text_augmentation and text_column in row:
                    augmented_text = self._augment_text(row[text_column])
                    if augmented_text:
                        augmented_row[text_column] = augmented_text
                
                augmented_data.append(augmented_row)
        
        augmented_df = pd.DataFrame(augmented_data)
        
        logger.info(f"Augmented dataset from {len(df)} to {len(augmented_df)} samples")
        
        return augmented_df
    
    def _enumerate_smiles(self, smiles: str) -> Optional[str]:
        """Generate different SMILES representation of the same molecule."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Generate random SMILES enumeration
            enumerated = Chem.MolToSmiles(mol, doRandom=True)
            return enumerated if enumerated != smiles else None
            
        except Exception as e:
            logger.warning(f"Error enumerating SMILES {smiles}: {e}")
            return None
    
    def _augment_text(self, text: str) -> Optional[str]:
        """Augment text description."""
        if not text or len(text.strip()) == 0:
            return None
        
        # Simple text augmentation strategies
        augmentations = [
            self._synonym_replacement,
            self._sentence_shuffling,
            self._add_descriptive_words
        ]
        
        augmentation_func = random.choice(augmentations)
        return augmentation_func(text)
    
    def _synonym_replacement(self, text: str) -> str:
        """Replace words with synonyms (simplified implementation)."""
        # Molecular property synonyms
        synonyms = {
            'active': 'potent',
            'inactive': 'weak',
            'compound': 'molecule',
            'drug': 'pharmaceutical',
            'inhibitor': 'blocker',
            'agonist': 'activator',
            'high': 'elevated',
            'low': 'reduced',
            'selective': 'specific',
            'potent': 'strong'
        }
        
        words = text.split()
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?')
            if word_lower in synonyms and random.random() < 0.3:
                words[i] = word.replace(word_lower, synonyms[word_lower])
        
        return ' '.join(words)
    
    def _sentence_shuffling(self, text: str) -> str:
        """Shuffle sentences in text."""
        sentences = text.split('.')
        if len(sentences) > 2:
            random.shuffle(sentences)
            return '.'.join(sentences)
        return text
    
    def _add_descriptive_words(self, text: str) -> str:
        """Add descriptive molecular terms."""
        descriptors = [
            'bioactive', 'small molecule', 'chemical entity',
            'organic compound', 'therapeutic agent'
        ]
        
        if random.random() < 0.5:
            descriptor = random.choice(descriptors)
            return f"{descriptor} {text}"
        
        return text
    
    def scaffold_hop(self, smiles: str, max_attempts: int = 10) -> Optional[str]:
        """
        Perform scaffold hopping to generate similar molecules.
        
        Args:
            smiles: Input SMILES
            max_attempts: Maximum attempts to generate valid molecule
            
        Returns:
            Scaffold-hopped SMILES or None
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Extract scaffold
            scaffold_extractor = ScaffoldExtractor()
            scaffold_smiles = scaffold_extractor.get_murcko_scaffold(smiles)
            if not scaffold_smiles:
                return None
            
            # This is a simplified scaffold hopping
            # In practice, you would use more sophisticated methods
            # like using generative models or fragment libraries
            
            for attempt in range(max_attempts):
                try:
                    # Generate random modifications
                    modified_mol = self._modify_molecule(mol)
                    if modified_mol:
                        new_smiles = Chem.MolToSmiles(modified_mol)
                        new_scaffold = scaffold_extractor.get_murcko_scaffold(new_smiles)
                        
                        # Check if scaffold is similar but not identical
                        if new_scaffold and new_scaffold != scaffold_smiles:
                            similarity = scaffold_extractor.compute_scaffold_similarity(
                                scaffold_smiles, new_scaffold
                            )
                            if 0.6 < similarity < 0.9:  # Similar but not identical
                                return new_smiles
                
                except Exception as e:
                    continue
            
            return None
            
        except Exception as e:
            logger.warning(f"Error in scaffold hopping for {smiles}: {e}")
            return None
    
    def _modify_molecule(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Apply random modifications to molecule."""
        # This is a simplified implementation
        # Real scaffold hopping would be more sophisticated
        
        modifications = [
            self._add_methyl_group,
            self._remove_methyl_group,
            self._change_ring_size
        ]
        
        modification = random.choice(modifications)
        return modification(mol)
    
    def _add_methyl_group(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Add a methyl group to a random position."""
        # Simplified implementation
        try:
            # This would need more sophisticated implementation
            return mol
        except:
            return None
    
    def _remove_methyl_group(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Remove a methyl group."""
        # Simplified implementation
        try:
            return mol
        except:
            return None
    
    def _change_ring_size(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Change ring size."""
        # Simplified implementation
        try:
            return mol
        except:
            return None


class TextProcessor:
    """Specialized text processing for molecular descriptions."""
    
    def __init__(self):
        """Initialize text processor."""
        # Chemical terms and abbreviations
        self.chemical_terms = {
            'IC50': 'half maximal inhibitory concentration',
            'EC50': 'half maximal effective concentration',
            'Ki': 'inhibition constant',
            'Kd': 'dissociation constant',
            'MW': 'molecular weight',
            'LogP': 'partition coefficient',
            'TPSA': 'topological polar surface area',
            'HBD': 'hydrogen bond donor',
            'HBA': 'hydrogen bond acceptor'
        }
    
    def normalize_text(self, text: str) -> str:
        """Normalize molecular text descriptions."""
        if not text:
            return ""
        
        # Convert to lowercase
        normalized = text.lower()
        
        # Expand abbreviations
        for abbrev, full_form in self.chemical_terms.items():
            normalized = normalized.replace(abbrev.lower(), full_form)
        
        # Clean up whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def extract_properties(self, text: str) -> Dict[str, Any]:
        """Extract numerical properties from text."""
        import re
        
        properties = {}
        
        # Extract IC50 values
        ic50_pattern = r'ic50[:\s]*([0-9.]+)\s*([nuμm]?m)'
        matches = re.findall(ic50_pattern, text.lower())
        if matches:
            value, unit = matches[0]
            properties['ic50'] = float(value)
            properties['ic50_unit'] = unit
        
        # Extract molecular weight
        mw_pattern = r'(?:molecular weight|mw)[:\s]*([0-9.]+)'
        matches = re.findall(mw_pattern, text.lower())
        if matches:
            properties['molecular_weight'] = float(matches[0])
        
        # Extract LogP
        logp_pattern = r'logp[:\s]*([0-9.-]+)'
        matches = re.findall(logp_pattern, text.lower())
        if matches:
            properties['logp'] = float(matches[0])
        
        return properties


class QualityFilter:
    """Filter molecules based on quality criteria."""
    
    def __init__(self,
                 lipinski_rule: bool = True,
                 pains_filter: bool = True,
                 custom_filters: Optional[List] = None):
        """
        Initialize quality filter.
        
        Args:
            lipinski_rule: Apply Lipinski's rule of five
            pains_filter: Filter PAINS compounds
            custom_filters: Custom filter functions
        """
        self.lipinski_rule = lipinski_rule
        self.pains_filter = pains_filter
        self.custom_filters = custom_filters or []
    
    def filter_molecules(self, smiles_list: List[str]) -> List[bool]:
        """
        Filter molecules based on quality criteria.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Boolean mask indicating which molecules pass filters
        """
        mask = []
        
        for smiles in smiles_list:
            passed = True
            
            # Lipinski's rule of five
            if self.lipinski_rule:
                passed &= self._check_lipinski(smiles)
            
            # PAINS filter
            if self.pains_filter:
                passed &= self._check_pains(smiles)
            
            # Custom filters
            for filter_func in self.custom_filters:
                passed &= filter_func(smiles)
            
            mask.append(passed)
        
        return mask
    
    def _check_lipinski(self, smiles: str) -> bool:
        """Check Lipinski's rule of five."""
        try:
            props = MolecularUtils.compute_molecular_properties(smiles)
            if not props:
                return False
            
            # Lipinski's rule of five
            return (
                props.get('molecular_weight', 0) <= 500 and
                props.get('logp', 0) <= 5 and
                props.get('num_hbd', 0) <= 5 and
                props.get('num_hba', 0) <= 10
            )
            
        except Exception:
            return False
    
    def _check_pains(self, smiles: str) -> bool:
        """Check for PAINS (Pan-Assay Interference Compounds)."""
        try:
            # This would require PAINS substructure filters
            # For now, return True (simplified implementation)
            return True
            
        except Exception:
            return False