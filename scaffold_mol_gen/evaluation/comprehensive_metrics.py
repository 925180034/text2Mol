"""
Comprehensive Evaluation Metrics for Multi-Modal Molecular Generation

This module implements all required evaluation metrics for the 9-modality molecular generation system:
- Uniqueness: Measures the diversity of generated molecules
- Novelty: Evaluates how different generated molecules are from training data
- Validity: Assesses chemical validity of generated molecules
- BLEU: Text similarity metric for SMILES sequences
- Exact Match: Exact matching rate between generated and target molecules
- Levenshtein Distance: Edit distance between sequences
- FTS (Fingerprint Tanimoto Similarity): Multiple fingerprint-based similarities
- FCD (Fréchet ChemNet Distance): Distribution distance in chemical space
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import Counter
import logging
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import Levenshtein
import torch
import torch.nn as nn
from scipy import linalg

logger = logging.getLogger(__name__)

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class ComprehensiveMetrics:
    """
    Complete evaluation metrics suite for molecular generation
    
    Implements all 11 required metrics for evaluating generated molecules
    across different modalities and quality dimensions.
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize the comprehensive metrics calculator
        
        Args:
            device: Device for neural network computations (cuda/cpu)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.smoothing_function = SmoothingFunction().method1
        
    def calculate_uniqueness(self, molecules: List[str]) -> float:
        """
        Calculate uniqueness: percentage of unique molecules in generated set
        
        Uniqueness = |unique_molecules| / |all_molecules|
        
        Args:
            molecules: List of SMILES strings
            
        Returns:
            Uniqueness score (0-1)
        """
        if not molecules:
            return 0.0
            
        # Canonicalize SMILES for fair comparison
        canonical_smiles = []
        for smi in molecules:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                canonical = Chem.MolToSmiles(mol, canonical=True)
                canonical_smiles.append(canonical)
        
        if not canonical_smiles:
            return 0.0
            
        unique_molecules = set(canonical_smiles)
        uniqueness = len(unique_molecules) / len(canonical_smiles)
        
        return uniqueness
    
    def calculate_novelty(self, generated: List[str], reference: List[str]) -> float:
        """
        Calculate novelty: percentage of generated molecules not in reference set
        
        Novelty = |generated - reference| / |generated|
        
        Args:
            generated: List of generated SMILES
            reference: List of reference/training SMILES
            
        Returns:
            Novelty score (0-1)
        """
        if not generated:
            return 0.0
            
        # Canonicalize both sets
        gen_canonical = set()
        for smi in generated:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                gen_canonical.add(Chem.MolToSmiles(mol, canonical=True))
        
        ref_canonical = set()
        for smi in reference:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                ref_canonical.add(Chem.MolToSmiles(mol, canonical=True))
        
        if not gen_canonical:
            return 0.0
            
        novel_molecules = gen_canonical - ref_canonical
        novelty = len(novel_molecules) / len(gen_canonical)
        
        return novelty
    
    def calculate_validity(self, molecules: List[str]) -> float:
        """
        Calculate validity: percentage of chemically valid molecules
        
        Validity = |valid_molecules| / |all_molecules|
        
        Args:
            molecules: List of SMILES strings
            
        Returns:
            Validity score (0-1)
        """
        if not molecules:
            return 0.0
            
        valid_count = 0
        for smi in molecules:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    # Additional validity check
                    Chem.SanitizeMol(mol)
                    valid_count += 1
            except:
                continue
                
        validity = valid_count / len(molecules)
        return validity
    
    def calculate_bleu(self, generated: List[str], references: List[str], 
                      n_gram: int = 4) -> Dict[str, float]:
        """
        Calculate BLEU scores for SMILES sequences
        
        Args:
            generated: List of generated SMILES
            references: List of reference SMILES
            n_gram: Maximum n-gram to consider
            
        Returns:
            Dictionary with BLEU-1 to BLEU-n scores
        """
        if not generated or not references:
            return {f'bleu_{i}': 0.0 for i in range(1, n_gram + 1)}
        
        bleu_scores = {f'bleu_{i}': [] for i in range(1, n_gram + 1)}
        
        for gen, ref in zip(generated, references):
            # Tokenize SMILES into characters
            gen_tokens = list(gen)
            ref_tokens = list(ref)
            
            # Calculate BLEU for different n-grams
            for n in range(1, min(n_gram + 1, len(gen_tokens) + 1)):
                weights = [1.0/n] * n + [0.0] * (4 - n)
                score = sentence_bleu([ref_tokens], gen_tokens, 
                                    weights=weights, 
                                    smoothing_function=self.smoothing_function)
                bleu_scores[f'bleu_{n}'].append(score)
        
        # Average scores
        avg_scores = {}
        for key, scores in bleu_scores.items():
            avg_scores[key] = np.mean(scores) if scores else 0.0
            
        return avg_scores
    
    def calculate_exact_match(self, generated: List[str], targets: List[str]) -> float:
        """
        Calculate exact match rate between generated and target molecules
        
        Exact = |exact_matches| / |comparisons|
        
        Args:
            generated: List of generated SMILES
            targets: List of target SMILES
            
        Returns:
            Exact match rate (0-1)
        """
        if not generated or not targets:
            return 0.0
            
        exact_matches = 0
        comparisons = min(len(generated), len(targets))
        
        for gen, target in zip(generated[:comparisons], targets[:comparisons]):
            # Canonicalize for fair comparison
            gen_mol = Chem.MolFromSmiles(gen)
            target_mol = Chem.MolFromSmiles(target)
            
            if gen_mol is not None and target_mol is not None:
                gen_canonical = Chem.MolToSmiles(gen_mol, canonical=True)
                target_canonical = Chem.MolToSmiles(target_mol, canonical=True)
                
                if gen_canonical == target_canonical:
                    exact_matches += 1
                    
        exact_match_rate = exact_matches / comparisons if comparisons > 0 else 0.0
        return exact_match_rate
    
    def calculate_levenshtein(self, generated: List[str], targets: List[str]) -> Dict[str, float]:
        """
        Calculate Levenshtein (edit) distance between sequences
        
        Args:
            generated: List of generated SMILES
            targets: List of target SMILES
            
        Returns:
            Dictionary with mean, std, and normalized Levenshtein distances
        """
        if not generated or not targets:
            return {'levenshtein_mean': 0.0, 'levenshtein_std': 0.0, 
                   'levenshtein_normalized': 0.0}
        
        distances = []
        normalized_distances = []
        
        for gen, target in zip(generated, targets):
            # Calculate raw Levenshtein distance
            distance = Levenshtein.distance(gen, target)
            distances.append(distance)
            
            # Normalized by maximum possible distance
            max_len = max(len(gen), len(target))
            if max_len > 0:
                normalized_distances.append(distance / max_len)
            else:
                normalized_distances.append(0.0)
        
        return {
            'levenshtein_mean': np.mean(distances),
            'levenshtein_std': np.std(distances),
            'levenshtein_normalized': np.mean(normalized_distances)
        }
    
    def calculate_fingerprint_similarity(self, generated: List[str], targets: List[str],
                                        fp_type: str = 'all') -> Dict[str, float]:
        """
        Calculate Fingerprint Tanimoto Similarity (FTS) using different fingerprints
        
        Args:
            generated: List of generated SMILES
            targets: List of target SMILES
            fp_type: Type of fingerprint ('maccs', 'morgan', 'rdk', 'all')
            
        Returns:
            Dictionary with Tanimoto similarities for each fingerprint type
        """
        similarities = {}
        
        if fp_type in ['maccs', 'all']:
            similarities['maccs_fts'] = self._calculate_maccs_similarity(generated, targets)
            
        if fp_type in ['morgan', 'all']:
            similarities['morgan_fts'] = self._calculate_morgan_similarity(generated, targets)
            
        if fp_type in ['rdk', 'all']:
            similarities['rdk_fts'] = self._calculate_rdk_similarity(generated, targets)
            
        return similarities
    
    def _calculate_maccs_similarity(self, generated: List[str], targets: List[str]) -> float:
        """Calculate MACCS fingerprint Tanimoto similarity"""
        similarities = []
        
        for gen, target in zip(generated, targets):
            gen_mol = Chem.MolFromSmiles(gen)
            target_mol = Chem.MolFromSmiles(target)
            
            if gen_mol is not None and target_mol is not None:
                try:
                    gen_fp = MACCSkeys.GenMACCSKeys(gen_mol)
                    target_fp = MACCSkeys.GenMACCSKeys(target_mol)
                    similarity = DataStructs.TanimotoSimilarity(gen_fp, target_fp)
                    similarities.append(similarity)
                except:
                    continue
                    
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_morgan_similarity(self, generated: List[str], targets: List[str]) -> float:
        """Calculate Morgan fingerprint Tanimoto similarity"""
        similarities = []
        
        for gen, target in zip(generated, targets):
            gen_mol = Chem.MolFromSmiles(gen)
            target_mol = Chem.MolFromSmiles(target)
            
            if gen_mol is not None and target_mol is not None:
                try:
                    gen_fp = AllChem.GetMorganFingerprintAsBitVect(gen_mol, radius=2, nBits=2048)
                    target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, radius=2, nBits=2048)
                    similarity = DataStructs.TanimotoSimilarity(gen_fp, target_fp)
                    similarities.append(similarity)
                except:
                    continue
                    
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_rdk_similarity(self, generated: List[str], targets: List[str]) -> float:
        """Calculate RDKit fingerprint Tanimoto similarity"""
        similarities = []
        
        for gen, target in zip(generated, targets):
            gen_mol = Chem.MolFromSmiles(gen)
            target_mol = Chem.MolFromSmiles(target)
            
            if gen_mol is not None and target_mol is not None:
                try:
                    gen_fp = Chem.RDKFingerprint(gen_mol)
                    target_fp = Chem.RDKFingerprint(target_mol)
                    similarity = DataStructs.TanimotoSimilarity(gen_fp, target_fp)
                    similarities.append(similarity)
                except:
                    continue
                    
        return np.mean(similarities) if similarities else 0.0
    
    def calculate_fcd(self, generated: List[str], reference: List[str],
                     use_pretrained: bool = False) -> float:
        """
        Calculate Fréchet ChemNet Distance (FCD)
        
        FCD measures the distance between distributions of generated and reference molecules
        in the chemical feature space.
        
        Args:
            generated: List of generated SMILES
            reference: List of reference SMILES
            use_pretrained: Whether to use pretrained ChemNet features
            
        Returns:
            FCD score (lower is better)
        """
        # For now, use a simplified version based on molecular descriptors
        # In production, this should use pretrained ChemNet features
        
        gen_features = self._extract_molecular_features(generated)
        ref_features = self._extract_molecular_features(reference)
        
        if gen_features is None or ref_features is None:
            return float('inf')
        
        # Calculate mean and covariance
        mu_gen = np.mean(gen_features, axis=0)
        mu_ref = np.mean(ref_features, axis=0)
        
        sigma_gen = np.cov(gen_features, rowvar=False)
        sigma_ref = np.cov(ref_features, rowvar=False)
        
        # Calculate FCD
        fcd = self._calculate_frechet_distance(mu_gen, sigma_gen, mu_ref, sigma_ref)
        
        return fcd
    
    def _extract_molecular_features(self, smiles_list: List[str]) -> Optional[np.ndarray]:
        """Extract molecular descriptor features for FCD calculation"""
        features = []
        
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                try:
                    # Use a subset of molecular descriptors
                    feature_vec = [
                        Descriptors.MolWt(mol),
                        Descriptors.MolLogP(mol),
                        Descriptors.NumHDonors(mol),
                        Descriptors.NumHAcceptors(mol),
                        Descriptors.NumRotatableBonds(mol),
                        Descriptors.TPSA(mol),
                        Descriptors.NumAromaticRings(mol),
                        Descriptors.NumSaturatedRings(mol),
                        Descriptors.NumAliphaticRings(mol),
                        Descriptors.RingCount(mol)
                    ]
                    features.append(feature_vec)
                except:
                    continue
                    
        if not features:
            return None
            
        return np.array(features)
    
    def _calculate_frechet_distance(self, mu1: np.ndarray, sigma1: np.ndarray,
                                   mu2: np.ndarray, sigma2: np.ndarray) -> float:
        """Calculate Fréchet distance between two multivariate Gaussians"""
        try:
            # Ensure symmetric positive semi-definite
            sigma1 = (sigma1 + sigma1.T) / 2
            sigma2 = (sigma2 + sigma2.T) / 2
            
            # Add small epsilon for numerical stability
            eps = 1e-6
            sigma1 += eps * np.eye(sigma1.shape[0])
            sigma2 += eps * np.eye(sigma2.shape[0])
            
            # Calculate distance
            diff = mu1 - mu2
            
            # Product of covariances
            covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
            
            # Numerical stability
            if not np.isfinite(covmean).all():
                offset = np.eye(sigma1.shape[0]) * eps
                covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
            
            # Ensure real
            if np.iscomplexobj(covmean):
                covmean = covmean.real
                
            tr_covmean = np.trace(covmean)
            
            fcd = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
            
            return float(fcd)
            
        except:
            return float('inf')
    
    def calculate_all_metrics(self, generated: List[str], targets: List[str],
                            reference: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Calculate all evaluation metrics at once
        
        Args:
            generated: List of generated SMILES
            targets: List of target SMILES
            reference: Optional list of reference/training SMILES for novelty and FCD
            
        Returns:
            Dictionary containing all metric scores
        """
        metrics = {}
        
        # Basic quality metrics
        metrics['uniqueness'] = self.calculate_uniqueness(generated)
        metrics['validity'] = self.calculate_validity(generated)
        
        # Novelty (use targets as reference if not provided)
        if reference is None:
            reference = targets
        metrics['novelty'] = self.calculate_novelty(generated, reference)
        
        # Sequence similarity metrics
        bleu_scores = self.calculate_bleu(generated, targets)
        metrics.update(bleu_scores)
        
        # Exact match
        metrics['exact_match'] = self.calculate_exact_match(generated, targets)
        
        # Edit distance
        lev_scores = self.calculate_levenshtein(generated, targets)
        metrics.update(lev_scores)
        
        # Fingerprint similarities
        fp_similarities = self.calculate_fingerprint_similarity(generated, targets, fp_type='all')
        metrics.update(fp_similarities)
        
        # FCD
        metrics['fcd'] = self.calculate_fcd(generated, reference)
        
        return metrics
    
    def format_metrics_report(self, metrics: Dict[str, float]) -> str:
        """
        Format metrics into a readable report
        
        Args:
            metrics: Dictionary of metric scores
            
        Returns:
            Formatted string report
        """
        report = "=" * 60 + "\n"
        report += "COMPREHENSIVE EVALUATION METRICS REPORT\n"
        report += "=" * 60 + "\n\n"
        
        # Quality metrics
        report += "📊 QUALITY METRICS\n"
        report += "-" * 30 + "\n"
        report += f"Validity:    {metrics.get('validity', 0):.2%}\n"
        report += f"Uniqueness:  {metrics.get('uniqueness', 0):.2%}\n"
        report += f"Novelty:     {metrics.get('novelty', 0):.2%}\n\n"
        
        # Similarity metrics
        report += "🔍 SIMILARITY METRICS\n"
        report += "-" * 30 + "\n"
        report += f"Exact Match: {metrics.get('exact_match', 0):.2%}\n"
        report += f"BLEU-1:      {metrics.get('bleu_1', 0):.3f}\n"
        report += f"BLEU-2:      {metrics.get('bleu_2', 0):.3f}\n"
        report += f"BLEU-3:      {metrics.get('bleu_3', 0):.3f}\n"
        report += f"BLEU-4:      {metrics.get('bleu_4', 0):.3f}\n\n"
        
        # Edit distance
        report += "✏️ EDIT DISTANCE\n"
        report += "-" * 30 + "\n"
        report += f"Levenshtein (mean):       {metrics.get('levenshtein_mean', 0):.2f}\n"
        report += f"Levenshtein (normalized): {metrics.get('levenshtein_normalized', 0):.3f}\n\n"
        
        # Fingerprint similarities
        report += "🧬 FINGERPRINT SIMILARITIES\n"
        report += "-" * 30 + "\n"
        report += f"MACCS FTS:   {metrics.get('maccs_fts', 0):.3f}\n"
        report += f"Morgan FTS:  {metrics.get('morgan_fts', 0):.3f}\n"
        report += f"RDK FTS:     {metrics.get('rdk_fts', 0):.3f}\n\n"
        
        # Distribution distance
        report += "📈 DISTRIBUTION METRICS\n"
        report += "-" * 30 + "\n"
        report += f"FCD Score:   {metrics.get('fcd', float('inf')):.3f}\n"
        report += "(Lower FCD is better)\n"
        
        report += "\n" + "=" * 60
        
        return report


# Example usage
if __name__ == "__main__":
    # Create metrics calculator
    metrics_calc = ComprehensiveMetrics()
    
    # Example molecules
    generated = [
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "CC1CC(C)C(O)C(C)C1N",
        "CC(=O)Oc1ccccc1C(=O)O"
    ]
    
    targets = [
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "CC1CC(C)C(O)C(C)C1N",
        "CC(=O)Oc1ccccc1C(=O)O"
    ]
    
    # Calculate all metrics
    all_metrics = metrics_calc.calculate_all_metrics(generated, targets)
    
    # Print formatted report
    report = metrics_calc.format_metrics_report(all_metrics)
    print(report)