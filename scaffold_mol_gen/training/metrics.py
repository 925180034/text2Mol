"""
Evaluation metrics for scaffold-based molecular generation.

This module provides comprehensive metrics for evaluating:
- Molecular validity and quality
- Scaffold preservation
- Generation diversity and novelty
- Text-molecule alignment
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import Counter, defaultdict
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from ..utils.mol_utils import MolecularUtils, compute_tanimoto_similarity
from ..utils.scaffold_utils import ScaffoldExtractor

logger = logging.getLogger(__name__)

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class MolecularMetrics:
    """Comprehensive molecular evaluation metrics."""
    
    def __init__(self):
        """Initialize molecular metrics calculator."""
        self.scaffold_extractor = ScaffoldExtractor()
        
    def compute_validity(self, smiles_list: List[str]) -> Dict[str, float]:
        """
        Compute validity metrics.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Dictionary with validity metrics
        """
        valid_count = 0
        total_count = len(smiles_list)
        
        for smiles in smiles_list:
            if MolecularUtils.validate_smiles(smiles):
                valid_count += 1
        
        validity = valid_count / total_count if total_count > 0 else 0.0
        
        return {
            'validity': validity,
            'valid_count': valid_count,
            'total_count': total_count
        }
    
    def compute_uniqueness(self, smiles_list: List[str]) -> Dict[str, float]:
        """
        Compute uniqueness metrics.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Dictionary with uniqueness metrics
        """
        # Filter valid SMILES and canonicalize
        canonical_smiles = set()
        valid_smiles = []
        
        for smiles in smiles_list:
            if MolecularUtils.validate_smiles(smiles):
                canonical = MolecularUtils.canonicalize_smiles(smiles)
                if canonical:
                    valid_smiles.append(canonical)
                    canonical_smiles.add(canonical)
        
        unique_count = len(canonical_smiles)
        valid_count = len(valid_smiles)
        
        uniqueness = unique_count / valid_count if valid_count > 0 else 0.0
        
        return {
            'uniqueness': uniqueness,
            'unique_count': unique_count,
            'valid_count': valid_count,
            'duplicate_count': valid_count - unique_count
        }
    
    def compute_novelty(self, generated_smiles: List[str], 
                       reference_smiles: List[str]) -> Dict[str, float]:
        """
        Compute novelty metrics (molecules not in reference set).
        
        Args:
            generated_smiles: Generated SMILES strings
            reference_smiles: Reference/training SMILES strings
            
        Returns:
            Dictionary with novelty metrics
        """
        # Canonicalize reference set
        reference_set = set()
        for smiles in reference_smiles:
            canonical = MolecularUtils.canonicalize_smiles(smiles)
            if canonical:
                reference_set.add(canonical)
        
        # Check novelty of generated molecules
        novel_count = 0
        valid_generated = []
        
        for smiles in generated_smiles:
            if MolecularUtils.validate_smiles(smiles):
                canonical = MolecularUtils.canonicalize_smiles(smiles)
                if canonical:
                    valid_generated.append(canonical)
                    if canonical not in reference_set:
                        novel_count += 1
        
        valid_count = len(valid_generated)
        novelty = novel_count / valid_count if valid_count > 0 else 0.0
        
        return {
            'novelty': novelty,
            'novel_count': novel_count,
            'valid_generated_count': valid_count,
            'reference_set_size': len(reference_set)
        }
    
    def compute_diversity(self, smiles_list: List[str], 
                         similarity_threshold: float = 0.7) -> Dict[str, float]:
        """
        Compute diversity metrics based on molecular similarity.
        
        Args:
            smiles_list: List of SMILES strings
            similarity_threshold: Threshold for considering molecules similar
            
        Returns:
            Dictionary with diversity metrics
        """
        # Filter valid SMILES
        valid_smiles = [s for s in smiles_list if MolecularUtils.validate_smiles(s)]
        
        if len(valid_smiles) < 2:
            return {
                'mean_pairwise_tanimoto': 0.0,
                'diversity_score': 1.0,
                'cluster_count': len(valid_smiles),
                'valid_count': len(valid_smiles)
            }
        
        # Compute pairwise similarities
        similarities = []
        for i in range(len(valid_smiles)):
            for j in range(i + 1, len(valid_smiles)):
                sim = compute_tanimoto_similarity(valid_smiles[i], valid_smiles[j])
                similarities.append(sim)
        
        mean_similarity = np.mean(similarities) if similarities else 0.0
        diversity_score = 1.0 - mean_similarity
        
        # Estimate number of clusters (molecules with similarity < threshold)
        cluster_count = self._estimate_clusters(valid_smiles, similarity_threshold)
        
        return {
            'mean_pairwise_tanimoto': mean_similarity,
            'diversity_score': diversity_score,
            'cluster_count': cluster_count,
            'valid_count': len(valid_smiles)
        }
    
    def _estimate_clusters(self, smiles_list: List[str], 
                          threshold: float) -> int:
        """Estimate number of molecular clusters."""
        clusters = []
        
        for smiles in smiles_list:
            # Find if molecule belongs to existing cluster
            assigned = False
            for cluster in clusters:
                representative = cluster[0]
                similarity = compute_tanimoto_similarity(smiles, representative)
                if similarity > threshold:
                    cluster.append(smiles)
                    assigned = True
                    break
            
            # Create new cluster if not assigned
            if not assigned:
                clusters.append([smiles])
        
        return len(clusters)
    
    def compute_drug_likeness(self, smiles_list: List[str]) -> Dict[str, float]:
        """
        Compute drug-likeness metrics (Lipinski's Rule of Five).
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Dictionary with drug-likeness metrics
        """
        lipinski_compliant = 0
        valid_molecules = 0
        property_stats = defaultdict(list)
        
        for smiles in smiles_list:
            if not MolecularUtils.validate_smiles(smiles):
                continue
                
            valid_molecules += 1
            props = MolecularUtils.compute_molecular_properties(smiles)
            
            if not props:
                continue
            
            # Collect properties
            for prop, value in props.items():
                property_stats[prop].append(value)
            
            # Check Lipinski's Rule of Five
            mw = props.get('molecular_weight', 0)
            logp = props.get('logp', 0)
            hbd = props.get('num_hbd', 0)
            hba = props.get('num_hba', 0)
            
            if mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10:
                lipinski_compliant += 1
        
        # Compute statistics
        lipinski_ratio = lipinski_compliant / valid_molecules if valid_molecules > 0 else 0.0
        
        stats = {
            'lipinski_compliance': lipinski_ratio,
            'lipinski_compliant_count': lipinski_compliant,
            'valid_molecules': valid_molecules
        }
        
        # Add property statistics
        for prop, values in property_stats.items():
            if values:
                stats[f'mean_{prop}'] = np.mean(values)
                stats[f'std_{prop}'] = np.std(values)
        
        return stats
    
    def compute_scaffold_metrics(self, generated_smiles: List[str],
                                target_scaffolds: List[str]) -> Dict[str, float]:
        """
        Compute scaffold-related metrics.
        
        Args:
            generated_smiles: Generated SMILES strings
            target_scaffolds: Target scaffold SMILES strings
            
        Returns:
            Dictionary with scaffold metrics
        """
        if len(generated_smiles) != len(target_scaffolds):
            logger.warning("Mismatch in generated and target scaffold counts")
            return {}
        
        scaffold_preserved = 0
        scaffold_similar = 0
        valid_pairs = 0
        similarities = []
        
        for gen_smiles, target_scaffold in zip(generated_smiles, target_scaffolds):
            if not MolecularUtils.validate_smiles(gen_smiles):
                continue
                
            valid_pairs += 1
            
            # Extract scaffold from generated molecule
            gen_scaffold = self.scaffold_extractor.get_murcko_scaffold(gen_smiles)
            
            if not gen_scaffold or not target_scaffold:
                continue
            
            # Exact match
            if gen_scaffold == target_scaffold:
                scaffold_preserved += 1
                scaffold_similar += 1
                similarities.append(1.0)
            else:
                # Similarity match
                similarity = self.scaffold_extractor.compute_scaffold_similarity(
                    gen_scaffold, target_scaffold
                )
                similarities.append(similarity)
                
                if similarity > 0.8:  # High similarity threshold
                    scaffold_similar += 1
        
        preservation_rate = scaffold_preserved / valid_pairs if valid_pairs > 0 else 0.0
        similarity_rate = scaffold_similar / valid_pairs if valid_pairs > 0 else 0.0
        mean_similarity = np.mean(similarities) if similarities else 0.0
        
        return {
            'scaffold_preservation_rate': preservation_rate,
            'scaffold_similarity_rate': similarity_rate,  
            'mean_scaffold_similarity': mean_similarity,
            'scaffold_preserved_count': scaffold_preserved,
            'scaffold_similar_count': scaffold_similar,
            'valid_pairs': valid_pairs
        }


class GenerationMetrics:
    """Metrics for evaluating generation quality and text alignment."""
    
    def __init__(self):
        """Initialize generation metrics calculator."""
        self.molecular_metrics = MolecularMetrics()
        
    def compute_metrics(self, generated_smiles: List[str],
                       target_smiles: List[str],
                       generated_texts: Optional[List[str]] = None,
                       target_texts: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Compute comprehensive generation metrics.
        
        Args:
            generated_smiles: Generated SMILES strings
            target_smiles: Target SMILES strings  
            generated_texts: Generated text descriptions (optional)
            target_texts: Target text descriptions (optional)
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Molecular metrics
        validity_metrics = self.molecular_metrics.compute_validity(generated_smiles)
        uniqueness_metrics = self.molecular_metrics.compute_uniqueness(generated_smiles)
        novelty_metrics = self.molecular_metrics.compute_novelty(generated_smiles, target_smiles)
        diversity_metrics = self.molecular_metrics.compute_diversity(generated_smiles)
        drug_metrics = self.molecular_metrics.compute_drug_likeness(generated_smiles)
        
        metrics.update(validity_metrics)
        metrics.update(uniqueness_metrics)
        metrics.update(novelty_metrics)
        metrics.update(diversity_metrics)
        metrics.update(drug_metrics)
        
        # Molecule-specific metrics
        if target_smiles:
            molecular_similarity = self.compute_molecular_similarity(
                generated_smiles, target_smiles
            )
            metrics.update(molecular_similarity)
        
        # Text metrics
        if generated_texts and target_texts:
            text_metrics = self.compute_text_metrics(generated_texts, target_texts)
            metrics.update(text_metrics)
        
        return metrics
    
    def compute_molecular_similarity(self, generated_smiles: List[str],
                                   target_smiles: List[str]) -> Dict[str, float]:
        """
        Compute similarity between generated and target molecules.
        
        Args:
            generated_smiles: Generated SMILES strings
            target_smiles: Target SMILES strings
            
        Returns:
            Dictionary with similarity metrics
        """
        if len(generated_smiles) != len(target_smiles):
            logger.warning("Mismatch in generated and target molecule counts")
            return {}
        
        similarities = []
        valid_pairs = 0
        
        for gen, target in zip(generated_smiles, target_smiles):
            if (MolecularUtils.validate_smiles(gen) and 
                MolecularUtils.validate_smiles(target)):
                valid_pairs += 1
                similarity = compute_tanimoto_similarity(gen, target)
                similarities.append(similarity)
        
        if not similarities:
            return {
                'mean_tanimoto_similarity': 0.0,
                'std_tanimoto_similarity': 0.0,
                'valid_similarity_pairs': 0
            }
        
        return {
            'mean_tanimoto_similarity': np.mean(similarities),
            'std_tanimoto_similarity': np.std(similarities),
            'median_tanimoto_similarity': np.median(similarities),
            'min_tanimoto_similarity': np.min(similarities),
            'max_tanimoto_similarity': np.max(similarities),
            'valid_similarity_pairs': valid_pairs
        }
    
    def compute_text_metrics(self, generated_texts: List[str],
                           target_texts: List[str]) -> Dict[str, float]:
        """
        Compute text generation metrics.
        
        Args:
            generated_texts: Generated text descriptions
            target_texts: Target text descriptions
            
        Returns:
            Dictionary with text metrics
        """
        if len(generated_texts) != len(target_texts):
            logger.warning("Mismatch in generated and target text counts")
            return {}
        
        bleu_scores = []
        rouge_scores = []
        
        smoothing = SmoothingFunction().method1
        
        for gen_text, target_text in zip(generated_texts, target_texts):
            if not gen_text or not target_text:
                continue
            
            # Tokenize
            gen_tokens = nltk.word_tokenize(gen_text.lower())
            target_tokens = nltk.word_tokenize(target_text.lower())
            
            # BLEU score
            bleu = sentence_bleu([target_tokens], gen_tokens, smoothing_function=smoothing)
            bleu_scores.append(bleu)
            
            # Simple ROUGE-L approximation
            rouge = self._compute_rouge_l(gen_tokens, target_tokens)
            rouge_scores.append(rouge)
        
        metrics = {}
        
        if bleu_scores:
            metrics.update({
                'mean_bleu_score': np.mean(bleu_scores),
                'std_bleu_score': np.std(bleu_scores),
                'median_bleu_score': np.median(bleu_scores)
            })
        
        if rouge_scores:
            metrics.update({
                'mean_rouge_l': np.mean(rouge_scores),
                'std_rouge_l': np.std(rouge_scores),
                'median_rouge_l': np.median(rouge_scores)
            })
        
        return metrics
    
    def _compute_rouge_l(self, generated_tokens: List[str], 
                        target_tokens: List[str]) -> float:
        """Compute ROUGE-L score (longest common subsequence)."""
        if not generated_tokens or not target_tokens:
            return 0.0
        
        # Dynamic programming for LCS
        m, n = len(generated_tokens), len(target_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if generated_tokens[i-1] == target_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        
        # ROUGE-L = LCS / length
        precision = lcs_length / len(generated_tokens) if generated_tokens else 0
        recall = lcs_length / len(target_tokens) if target_tokens else 0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def compute_comprehensive_metrics(self, generated_smiles: List[str],
                                    target_smiles: List[str],
                                    target_scaffolds: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            generated_smiles: Generated SMILES strings
            target_smiles: Target SMILES strings
            target_scaffolds: Target scaffold SMILES (optional)
            
        Returns:
            Complete metrics dictionary
        """
        metrics = self.compute_metrics(generated_smiles, target_smiles)
        
        # Add scaffold-specific metrics if available
        if target_scaffolds:
            scaffold_metrics = self.molecular_metrics.compute_scaffold_metrics(
                generated_smiles, target_scaffolds
            )
            metrics.update(scaffold_metrics)
        
        # Compute additional quality metrics
        quality_metrics = self._compute_quality_metrics(generated_smiles, target_smiles)
        metrics.update(quality_metrics)
        
        return metrics
    
    def _compute_quality_metrics(self, generated_smiles: List[str],
                               target_smiles: List[str]) -> Dict[str, float]:
        """Compute additional quality metrics."""
        # Fréchet ChemNet Distance (FCD) placeholder
        # In practice, this would require a pre-trained ChemNet model
        fcd_score = 0.0  # Placeholder
        
        # Structure-Activity Similarity (SAS) score placeholder
        sas_score = 0.0  # Placeholder
        
        # Quantitative Estimate of Drug-likeness (QED) score
        qed_scores = []
        for smiles in generated_smiles:
            if MolecularUtils.validate_smiles(smiles):
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    # Simplified QED calculation
                    props = MolecularUtils.compute_molecular_properties(smiles)
                    if props:
                        # Simplified drug-likeness score
                        mw = props.get('molecular_weight', 500)
                        logp = props.get('logp', 3)
                        
                        # Simple scoring function
                        mw_score = 1.0 if 150 <= mw <= 500 else 0.5
                        logp_score = 1.0 if -2 <= logp <= 5 else 0.5
                        
                        qed = (mw_score + logp_score) / 2
                        qed_scores.append(qed)
        
        mean_qed = np.mean(qed_scores) if qed_scores else 0.0
        
        return {
            'fcd_score': fcd_score,
            'sas_score': sas_score,
            'mean_qed_score': mean_qed,
            'std_qed_score': np.std(qed_scores) if qed_scores else 0.0
        }


class BenchmarkMetrics:
    """Standardized benchmark metrics for molecular generation."""
    
    def __init__(self):
        """Initialize benchmark metrics."""
        self.generation_metrics = GenerationMetrics()
        
    def evaluate_benchmark(self, generated_smiles: List[str],
                          test_smiles: List[str],
                          train_smiles: List[str],
                          target_scaffolds: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate using standard benchmark metrics.
        
        Args:
            generated_smiles: Generated molecules
            test_smiles: Test set molecules  
            train_smiles: Training set molecules
            target_scaffolds: Target scaffolds (optional)
            
        Returns:
            Benchmark metrics dictionary
        """
        # Core metrics
        validity = self.generation_metrics.molecular_metrics.compute_validity(generated_smiles)
        uniqueness = self.generation_metrics.molecular_metrics.compute_uniqueness(generated_smiles) 
        novelty = self.generation_metrics.molecular_metrics.compute_novelty(generated_smiles, train_smiles)
        diversity = self.generation_metrics.molecular_metrics.compute_diversity(generated_smiles)
        
        # Combine into standard benchmark format
        benchmark_results = {
            'Validity': validity['validity'],
            'Uniqueness': uniqueness['uniqueness'], 
            'Novelty': novelty['novelty'],
            'Diversity': diversity['diversity_score'],
            'Drug_likeness': 0.0  # Placeholder
        }
        
        # Add scaffold metrics if available
        if target_scaffolds:
            scaffold_metrics = self.generation_metrics.molecular_metrics.compute_scaffold_metrics(
                generated_smiles, target_scaffolds
            )
            benchmark_results['Scaffold_preservation'] = scaffold_metrics.get('scaffold_preservation_rate', 0.0)
        
        # Compute drug-likeness
        drug_metrics = self.generation_metrics.molecular_metrics.compute_drug_likeness(generated_smiles)
        benchmark_results['Drug_likeness'] = drug_metrics.get('lipinski_compliance', 0.0)
        
        return benchmark_results
    
    def print_benchmark_results(self, results: Dict[str, float]):
        """Print benchmark results in standard format."""
        print("\n" + "="*50)
        print("BENCHMARK RESULTS")
        print("="*50)
        
        for metric, value in results.items():
            print(f"{metric:20s}: {value:.4f}")
        
        print("="*50)


def evaluate_model_outputs(predictions: List[str],
                          targets: List[str],
                          reference_data: Optional[List[str]] = None,
                          scaffold_data: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Comprehensive evaluation function for model outputs.
    
    Args:
        predictions: Model predictions (SMILES)
        targets: Target molecules (SMILES)
        reference_data: Reference dataset for novelty computation
        scaffold_data: Target scaffolds for scaffold metrics
        
    Returns:
        Complete evaluation results
    """
    metrics_calculator = GenerationMetrics()
    
    # Basic generation metrics
    results = metrics_calculator.compute_comprehensive_metrics(
        predictions, targets, scaffold_data
    )
    
    # Add benchmark metrics if reference data available
    if reference_data:
        benchmark = BenchmarkMetrics()
        benchmark_results = benchmark.evaluate_benchmark(
            predictions, targets, reference_data, scaffold_data
        )
        results.update(benchmark_results)
    
    return results