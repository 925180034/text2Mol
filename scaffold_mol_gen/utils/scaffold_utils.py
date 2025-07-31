"""
Scaffold extraction and validation utilities.

This module provides comprehensive tools for working with molecular scaffolds,
including extraction, validation, and analysis functions.
"""

import logging
from typing import Optional, List, Tuple, Dict, Set
from rdkit import Chem
from rdkit.Chem import Scaffolds, AllChem, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

class ScaffoldExtractor:
    """
    Comprehensive scaffold extraction and validation toolkit.
    
    This class provides various methods for extracting and working with
    molecular scaffolds, with emphasis on Murcko scaffolds.
    """
    
    def __init__(self):
        self.cache = {}  # Cache for expensive scaffold computations
        
    @staticmethod
    def get_murcko_scaffold(smiles: str, canonical: bool = True) -> Optional[str]:
        """
        Extract Murcko scaffold from SMILES string.
        
        Args:
            smiles: Input SMILES string
            canonical: Whether to return canonical SMILES
            
        Returns:
            Scaffold SMILES string or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
                
            scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold_mol is None or scaffold_mol.GetNumAtoms() == 0:
                return None
                
            scaffold_smiles = Chem.MolToSmiles(scaffold_mol, canonical=canonical)
            return scaffold_smiles
            
        except Exception as e:
            logger.warning(f"Error extracting Murcko scaffold from {smiles}: {e}")
            return None
    
    @staticmethod
    def get_bemis_murcko_scaffold(smiles: str, include_chirality: bool = False) -> Optional[str]:
        """
        Extract Bemis-Murcko scaffold (rings and linkers).
        
        Args:
            smiles: Input SMILES string
            include_chirality: Whether to preserve chirality information
            
        Returns:
            Bemis-Murcko scaffold SMILES or None
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
                
            # Use Murcko scaffold as base
            scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold_mol is None:
                return None
                
            # Get scaffold with linkers
            scaffold_smiles = Chem.MolToSmiles(
                scaffold_mol, 
                canonical=True,
                isomericSmiles=include_chirality
            )
            
            return scaffold_smiles
            
        except Exception as e:
            logger.warning(f"Error extracting Bemis-Murcko scaffold from {smiles}: {e}")
            return None
    
    @staticmethod
    def get_generic_scaffold(smiles: str, remove_stereo: bool = True) -> Optional[str]:
        """
        Extract generic scaffold by removing atom-specific information.
        
        Args:
            smiles: Input SMILES string
            remove_stereo: Whether to remove stereochemistry
            
        Returns:
            Generic scaffold SMILES or None
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Get Murcko scaffold
            scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold_mol is None:
                return None
            
            # Make generic (optional: could replace atoms with carbons)
            if remove_stereo:
                Chem.RemoveStereochemistry(scaffold_mol)
            
            generic_smiles = Chem.MolToSmiles(scaffold_mol, canonical=True)
            return generic_smiles
            
        except Exception as e:
            logger.warning(f"Error extracting generic scaffold from {smiles}: {e}")
            return None
    
    @staticmethod
    def validate_scaffold_subset(scaffold_smiles: str, molecule_smiles: str) -> bool:
        """
        Validate that scaffold is a substructure of the molecule.
        
        Args:
            scaffold_smiles: Scaffold SMILES string
            molecule_smiles: Complete molecule SMILES string
            
        Returns:
            True if scaffold is contained in molecule, False otherwise
        """
        try:
            scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)
            molecule_mol = Chem.MolFromSmiles(molecule_smiles)
            
            if scaffold_mol is None or molecule_mol is None:
                return False
            
            # Check substructure match
            return molecule_mol.HasSubstructMatch(scaffold_mol)
            
        except Exception as e:
            logger.warning(f"Error validating scaffold subset: {e}")
            return False
    
    @staticmethod
    def get_scaffold_attachment_points(scaffold_smiles: str, molecule_smiles: str) -> List[Tuple[int, int]]:
        """
        Get attachment points where scaffold connects to the rest of the molecule.
        
        Args:
            scaffold_smiles: Scaffold SMILES string
            molecule_smiles: Complete molecule SMILES string
            
        Returns:
            List of (scaffold_atom_idx, molecule_neighbor_idx) tuples
        """
        try:
            scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)
            molecule_mol = Chem.MolFromSmiles(molecule_smiles)
            
            if scaffold_mol is None or molecule_mol is None:
                return []
            
            # Find scaffold matches in molecule
            matches = molecule_mol.GetSubstructMatches(scaffold_mol)
            if not matches:
                return []
            
            # Use first match
            match = matches[0]
            attachment_points = []
            
            # Find atoms in scaffold that connect to non-scaffold atoms
            for scaffold_atom_idx, mol_atom_idx in enumerate(match):
                mol_atom = molecule_mol.GetAtomWithIdx(mol_atom_idx)
                
                for neighbor in mol_atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    
                    # If neighbor is not in scaffold, it's an attachment point
                    if neighbor_idx not in match:
                        attachment_points.append((scaffold_atom_idx, neighbor_idx))
            
            return attachment_points
            
        except Exception as e:
            logger.warning(f"Error finding attachment points: {e}")
            return []
    
    @staticmethod
    def get_scaffold_substituents(scaffold_smiles: str, molecule_smiles: str) -> List[str]:
        """
        Extract substituents attached to the scaffold.
        
        Args:
            scaffold_smiles: Scaffold SMILES string
            molecule_smiles: Complete molecule SMILES string
            
        Returns:
            List of substituent SMILES strings
        """
        try:
            scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)
            molecule_mol = Chem.MolFromSmiles(molecule_smiles)
            
            if scaffold_mol is None or molecule_mol is None:
                return []
            
            # Create a copy for modification
            mol_copy = Chem.Mol(molecule_mol)
            
            # Find scaffold match
            matches = mol_copy.GetSubstructMatches(scaffold_mol)
            if not matches:
                return []
            
            match = matches[0]
            
            # Remove scaffold atoms to get substituents
            atoms_to_remove = sorted(match, reverse=True)
            editable_mol = Chem.EditableMol(mol_copy)
            
            for atom_idx in atoms_to_remove:
                editable_mol.RemoveAtom(atom_idx)
            
            substituent_mol = editable_mol.GetMol()
            if substituent_mol is None:
                return []
            
            # Get connected components (individual substituents)
            fragments = Chem.GetMolFrags(substituent_mol, asMols=True)
            substituents = []
            
            for fragment in fragments:
                if fragment.GetNumAtoms() > 0:
                    substituent_smiles = Chem.MolToSmiles(fragment)
                    substituents.append(substituent_smiles)
            
            return substituents
            
        except Exception as e:
            logger.warning(f"Error extracting substituents: {e}")
            return []
    
    def compute_scaffold_similarity(self, scaffold1: str, scaffold2: str, 
                                  method: str = 'tanimoto') -> float:
        """
        Compute similarity between two scaffolds.
        
        Args:
            scaffold1: First scaffold SMILES
            scaffold2: Second scaffold SMILES
            method: Similarity method ('tanimoto', 'dice', 'cosine')
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            mol1 = Chem.MolFromSmiles(scaffold1)
            mol2 = Chem.MolFromSmiles(scaffold2)
            
            if mol1 is None or mol2 is None:
                return 0.0
            
            # Generate fingerprints
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
            
            # Compute similarity
            if method == 'tanimoto':
                from rdkit import DataStructs
                return DataStructs.TanimotoSimilarity(fp1, fp2)
            elif method == 'dice':
                from rdkit import DataStructs
                return DataStructs.DiceSimilarity(fp1, fp2)
            elif method == 'cosine':
                from rdkit import DataStructs
                return DataStructs.CosineSimilarity(fp1, fp2)
            else:
                raise ValueError(f"Unknown similarity method: {method}")
                
        except Exception as e:
            logger.warning(f"Error computing scaffold similarity: {e}")
            return 0.0
    
    @staticmethod
    def scaffold_statistics(smiles_list: List[str]) -> Dict[str, any]:
        """
        Compute statistics about scaffolds in a dataset.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Dictionary with scaffold statistics
        """
        scaffold_counts = defaultdict(int)
        scaffold_molecules = defaultdict(list)
        valid_molecules = 0
        
        for i, smiles in enumerate(smiles_list):
            scaffold = ScaffoldExtractor.get_murcko_scaffold(smiles)
            if scaffold:
                scaffold_counts[scaffold] += 1
                scaffold_molecules[scaffold].append(i)
                valid_molecules += 1
        
        # Compute statistics
        unique_scaffolds = len(scaffold_counts)
        most_common_scaffold = max(scaffold_counts.items(), key=lambda x: x[1]) if scaffold_counts else (None, 0)
        scaffold_diversity = unique_scaffolds / valid_molecules if valid_molecules > 0 else 0
        
        # Size distribution
        scaffold_sizes = []
        for scaffold in scaffold_counts.keys():
            mol = Chem.MolFromSmiles(scaffold)
            if mol:
                scaffold_sizes.append(mol.GetNumAtoms())
        
        stats = {
            'total_molecules': len(smiles_list),
            'valid_molecules': valid_molecules,
            'unique_scaffolds': unique_scaffolds,
            'scaffold_diversity': scaffold_diversity,
            'most_common_scaffold': most_common_scaffold[0],
            'most_common_count': most_common_scaffold[1],
            'avg_scaffold_size': np.mean(scaffold_sizes) if scaffold_sizes else 0,
            'median_scaffold_size': np.median(scaffold_sizes) if scaffold_sizes else 0,
            'scaffold_size_std': np.std(scaffold_sizes) if scaffold_sizes else 0,
            'scaffold_counts': dict(scaffold_counts),
            'scaffold_molecules': dict(scaffold_molecules)
        }
        
        return stats
    
    def batch_extract_scaffolds(self, smiles_list: List[str], 
                               scaffold_type: str = 'murcko',
                               use_cache: bool = True) -> List[Optional[str]]:
        """
        Extract scaffolds from a batch of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            scaffold_type: Type of scaffold ('murcko', 'bemis_murcko', 'generic')
            use_cache: Whether to use caching for repeated computations
            
        Returns:
            List of scaffold SMILES (None for invalid molecules)
        """
        scaffolds = []
        
        extraction_methods = {
            'murcko': self.get_murcko_scaffold,
            'bemis_murcko': self.get_bemis_murcko_scaffold,
            'generic': self.get_generic_scaffold
        }
        
        if scaffold_type not in extraction_methods:
            raise ValueError(f"Unknown scaffold type: {scaffold_type}")
        
        extract_func = extraction_methods[scaffold_type]
        
        for smiles in smiles_list:
            if use_cache and smiles in self.cache:
                scaffold = self.cache[smiles].get(scaffold_type)
            else:
                scaffold = extract_func(smiles)
                
                if use_cache:
                    if smiles not in self.cache:
                        self.cache[smiles] = {}
                    self.cache[smiles][scaffold_type] = scaffold
            
            scaffolds.append(scaffold)
        
        return scaffolds
    
    @staticmethod
    def is_valid_scaffold(scaffold_smiles: str, min_atoms: int = 3, 
                         min_rings: int = 1) -> bool:
        """
        Check if a scaffold meets validity criteria.
        
        Args:
            scaffold_smiles: Scaffold SMILES string
            min_atoms: Minimum number of atoms
            min_rings: Minimum number of rings
            
        Returns:
            True if scaffold is valid, False otherwise
        """
        try:
            mol = Chem.MolFromSmiles(scaffold_smiles)
            if mol is None:
                return False
            
            # Check minimum atoms
            if mol.GetNumAtoms() < min_atoms:
                return False
            
            # Check minimum rings
            ring_info = mol.GetRingInfo()
            if len(ring_info.AtomRings()) < min_rings:
                return False
            
            return True
            
        except Exception:
            return False
    
    def clear_cache(self):
        """Clear the scaffold computation cache."""
        self.cache.clear()
    
    def get_cache_size(self) -> int:
        """Get the current cache size."""
        return len(self.cache)