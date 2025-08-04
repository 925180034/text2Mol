"""
Model evaluation framework for scaffold-based molecular generation.

This module provides comprehensive evaluation tools for assessing
model performance, generation quality, and comparative analysis.
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..models.core_model import ScaffoldBasedMolT5Generator
from ..training.metrics import GenerationMetrics, BenchmarkMetrics, evaluate_model_outputs
from ..data.collate import create_data_loader
from ..utils.mol_utils import MolecularUtils
from ..utils.scaffold_utils import ScaffoldExtractor
from ..utils.visualization import MolecularVisualizer

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluation framework.
    """
    
    def __init__(self, 
                 model: ScaffoldBasedMolT5Generator,
                 device: Optional[torch.device] = None,
                 output_dir: str = 'evaluation_results'):
        """
        Initialize model evaluator.
        
        Args:
            model: Model to evaluate
            device: Evaluation device
            output_dir: Directory to save evaluation results
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize metrics and visualization
        self.generation_metrics = GenerationMetrics()
        self.benchmark_metrics = BenchmarkMetrics()
        self.scaffold_extractor = ScaffoldExtractor()
        self.visualizer = MolecularVisualizer()
        
        logger.info(f"ModelEvaluator initialized on device: {self.device}")
    
    def evaluate_dataset(self, 
                        dataloader: DataLoader,
                        num_samples: Optional[int] = None,
                        generation_config: Optional[Dict[str, Any]] = None,
                        save_results: bool = True) -> Dict[str, Any]:
        """
        Evaluate model on a dataset.
        
        Args:
            dataloader: Data loader for evaluation
            num_samples: Maximum number of samples to evaluate
            generation_config: Generation configuration
            save_results: Whether to save detailed results
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info("Starting dataset evaluation...")
        start_time = time.time()
        
        # Generation configuration
        gen_config = generation_config or {
            'num_samples': 1,
            'max_length': 200,
            'num_beams': 5,
            'temperature': 0.8
        }
        
        # Collect predictions and targets
        all_predictions = []
        all_targets = []
        all_scaffolds = []
        all_texts = []
        sample_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if num_samples and sample_count >= num_samples:
                    break
                
                # Move batch to device
                batch = self._move_to_device(batch)
                
                # Generate predictions
                try:
                    predictions = self.model(batch, mode='inference', **gen_config)
                    
                    if isinstance(predictions, list):
                        all_predictions.extend(predictions)
                    else:
                        all_predictions.append(predictions)
                    
                    # Collect targets and metadata
                    if 'raw_data' in batch:
                        targets = batch['raw_data']['smiles']
                        scaffolds = batch['raw_data']['scaffold']
                        texts = batch['raw_data']['text']
                        
                        all_targets.extend(targets)
                        all_scaffolds.extend(scaffolds)
                        all_texts.extend(texts)
                    
                    sample_count += len(targets) if 'raw_data' in batch else 1
                    
                    if batch_idx % 10 == 0:
                        logger.info(f"Processed {sample_count} samples...")
                
                except Exception as e:
                    logger.warning(f"Error processing batch {batch_idx}: {e}")
                    continue
        
        evaluation_time = time.time() - start_time
        
        # Compute comprehensive metrics
        results = self._compute_comprehensive_metrics(
            all_predictions, all_targets, all_scaffolds, all_texts
        )
        
        # Add evaluation metadata
        results['evaluation_metadata'] = {
            'num_samples_evaluated': sample_count,
            'evaluation_time_seconds': evaluation_time,
            'generation_config': gen_config,
            'device': str(self.device)
        }
        
        # Save results if requested
        if save_results:
            self._save_evaluation_results(results, all_predictions, all_targets)
        
        logger.info(f"Dataset evaluation completed in {evaluation_time:.2f}s")
        logger.info(f"Evaluated {sample_count} samples")
        
        return results
    
    def evaluate_generation_quality(self,
                                  scaffold: str,
                                  text: str,
                                  num_samples: int = 10,
                                  generation_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate generation quality for a single input.
        
        Args:
            scaffold: Input scaffold SMILES
            text: Input text description
            num_samples: Number of samples to generate
            generation_config: Generation configuration
            
        Returns:
            Generation quality metrics
        """
        logger.info(f"Evaluating generation quality for scaffold: {scaffold}")
        
        # Generate multiple samples
        gen_config = generation_config or {}
        generated_samples = []
        
        with torch.no_grad():
            for i in range(num_samples):
                try:
                    samples = self.model.generate(
                        scaffold=scaffold,
                        text=text,
                        num_samples=1,
                        **gen_config
                    )
                    generated_samples.extend(samples)
                except Exception as e:
                    logger.warning(f"Generation error for sample {i}: {e}")
        
        if not generated_samples:
            return {'error': 'No valid samples generated'}
        
        # Compute quality metrics
        validity_metrics = self.generation_metrics.molecular_metrics.compute_validity(generated_samples)
        uniqueness_metrics = self.generation_metrics.molecular_metrics.compute_uniqueness(generated_samples)
        diversity_metrics = self.generation_metrics.molecular_metrics.compute_diversity(generated_samples)
        
        # Scaffold preservation
        target_scaffolds = [scaffold] * len(generated_samples)
        scaffold_metrics = self.generation_metrics.molecular_metrics.compute_scaffold_metrics(
            generated_samples, target_scaffolds
        )
        
        results = {
            'input_scaffold': scaffold,
            'input_text': text,
            'generated_samples': generated_samples,
            'num_generated': len(generated_samples),
            **validity_metrics,
            **uniqueness_metrics,
            **diversity_metrics,
            **scaffold_metrics
        }
        
        return results
    
    def comparative_evaluation(self,
                             baseline_predictions: List[str],
                             our_predictions: List[str],
                             targets: List[str],
                             scaffolds: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare model performance against baseline.
        
        Args:
            baseline_predictions: Baseline model predictions
            our_predictions: Our model predictions
            targets: Target molecules
            scaffolds: Target scaffolds (optional)
            
        Returns:
            Comparative evaluation results
        """
        logger.info("Running comparative evaluation...")
        
        # Evaluate both models
        baseline_results = evaluate_model_outputs(baseline_predictions, targets, scaffolds=scaffolds)
        our_results = evaluate_model_outputs(our_predictions, targets, scaffolds=scaffolds)
        
        # Compute improvements
        improvements = {}
        for metric in baseline_results.keys():
            if metric in our_results and isinstance(baseline_results[metric], (int, float)):
                baseline_val = baseline_results[metric]
                our_val = our_results[metric]
                
                if baseline_val != 0:
                    improvement = (our_val - baseline_val) / baseline_val * 100
                    improvements[f'{metric}_improvement_percent'] = improvement
                else:
                    improvements[f'{metric}_improvement_percent'] = 0.0
        
        # Statistical significance testing
        significance_results = self._statistical_significance_testing(
            baseline_predictions, our_predictions, targets
        )
        
        results = {
            'baseline_results': baseline_results,
            'our_results': our_results,
            'improvements': improvements,
            'statistical_significance': significance_results
        }
        
        return results
    
    def ablation_study(self,
                      dataloader: DataLoader,
                      components_to_ablate: List[str],
                      num_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform ablation study on model components.
        
        Args:
            dataloader: Evaluation data loader
            components_to_ablate: List of components to ablate
            num_samples: Number of samples to evaluate
            
        Returns:
            Ablation study results
        """
        logger.info(f"Running ablation study on components: {components_to_ablate}")
        
        # Baseline performance (full model)
        baseline_results = self.evaluate_dataset(
            dataloader, num_samples, save_results=False
        )
        
        ablation_results = {'baseline': baseline_results}
        
        # Ablate each component
        for component in components_to_ablate:
            logger.info(f"Ablating component: {component}")
            
            try:
                # Temporarily disable component
                self._disable_component(component)
                
                # Evaluate with component disabled
                ablated_results = self.evaluate_dataset(
                    dataloader, num_samples, save_results=False
                )
                
                ablation_results[f'without_{component}'] = ablated_results
                
                # Re-enable component
                self._enable_component(component)
                
            except Exception as e:
                logger.warning(f"Error ablating {component}: {e}")
                ablation_results[f'without_{component}'] = {'error': str(e)}
        
        # Compute impact of each component
        impact_analysis = self._analyze_component_impact(ablation_results)
        ablation_results['impact_analysis'] = impact_analysis
        
        return ablation_results
    
    def _compute_comprehensive_metrics(self,
                                     predictions: List[str],
                                     targets: List[str],
                                     scaffolds: List[str],
                                     texts: List[str]) -> Dict[str, Any]:
        """Compute comprehensive evaluation metrics with Phase 1 enhancements."""
        logger.info("Computing comprehensive metrics with Phase 1 enhancements...")
        
        # Use the enhanced GenerationMetrics for comprehensive evaluation
        results = self.generation_metrics.compute_comprehensive_metrics(
            generated_smiles=predictions,
            target_smiles=targets,
            reference_smiles=targets,  # Using targets as reference for basic similarity
            scaffold_data=scaffolds
        )
        
        # Add benchmark metrics for larger sample sizes
        if len(predictions) > 10:
            try:
                benchmark_results = self.benchmark_metrics.evaluate_benchmark(
                    predictions, targets, targets, scaffolds
                )
                results.update(benchmark_results)
            except Exception as e:
                logger.warning(f"Benchmark metrics computation failed: {e}")
        
        # Add scaffold-specific analysis
        if scaffolds and all(scaffold for scaffold in scaffolds):
            scaffold_analysis = self._compute_scaffold_analysis(predictions, targets, scaffolds)
            results.update(scaffold_analysis)
        
        # Add evaluation metadata
        results['enhanced_evaluation'] = {
            'phase1_metrics_included': True,
            'total_samples': len(predictions),
            'valid_predictions': sum(1 for p in predictions if self._is_valid_smiles(p)),
            'valid_targets': sum(1 for t in targets if self._is_valid_smiles(t))
        }
        
        return results
    
    def _compute_scaffold_analysis(self,
                                 predictions: List[str],
                                 targets: List[str],
                                 scaffolds: List[str]) -> Dict[str, Any]:
        """Compute detailed scaffold preservation analysis."""
        try:
            # Extract scaffolds from predictions
            pred_scaffolds = []
            for pred in predictions:
                if self._is_valid_smiles(pred):
                    pred_scaffold = self.scaffold_extractor.extract_scaffold(pred)
                    pred_scaffolds.append(pred_scaffold or '')
                else:
                    pred_scaffolds.append('')
            
            # Compute scaffold preservation rates
            exact_scaffold_matches = sum(
                1 for pred_scaffold, target_scaffold in zip(pred_scaffolds, scaffolds)
                if pred_scaffold and target_scaffold and pred_scaffold == target_scaffold
            )
            
            valid_scaffold_pairs = sum(
                1 for pred_scaffold, target_scaffold in zip(pred_scaffolds, scaffolds)
                if pred_scaffold and target_scaffold
            )
            
            scaffold_preservation_rate = (
                exact_scaffold_matches / valid_scaffold_pairs
                if valid_scaffold_pairs > 0 else 0.0
            )
            
            return {
                'scaffold_analysis': {
                    'exact_scaffold_matches': exact_scaffold_matches,
                    'valid_scaffold_pairs': valid_scaffold_pairs,
                    'scaffold_preservation_rate': scaffold_preservation_rate,
                    'scaffold_extraction_success_rate': sum(1 for s in pred_scaffolds if s) / len(pred_scaffolds)
                }
            }
            
        except Exception as e:
            logger.warning(f"Scaffold analysis failed: {e}")
            return {'scaffold_analysis': {'error': str(e)}}
    
    def _is_valid_smiles(self, smiles: str) -> bool:
        """Check if SMILES string is valid."""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    def _statistical_significance_testing(self,
                                        baseline_predictions: List[str],
                                        our_predictions: List[str],
                                        targets: List[str]) -> Dict[str, Any]:
        """Perform statistical significance testing."""
        try:
            from scipy import stats
        except ImportError:
            logger.warning("SciPy not available for statistical testing")
            return {'error': 'SciPy not available'}
        
        # Compute pairwise similarities for statistical testing
        baseline_similarities = []
        our_similarities = []
        
        for baseline, ours, target in zip(baseline_predictions, our_predictions, targets):
            if (MolecularUtils.validate_smiles(baseline) and 
                MolecularUtils.validate_smiles(ours) and
                MolecularUtils.validate_smiles(target)):
                
                baseline_sim = compute_tanimoto_similarity(baseline, target)
                our_sim = compute_tanimoto_similarity(ours, target)
                
                baseline_similarities.append(baseline_sim)
                our_similarities.append(our_sim)
        
        if len(baseline_similarities) < 5:
            return {'error': 'Insufficient valid samples for statistical testing'}
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(our_similarities, baseline_similarities)
        
        # Wilcoxon signed-rank test (non-parametric)
        w_stat, w_p_value = stats.wilcoxon(our_similarities, baseline_similarities)
        
        return {
            'paired_t_test': {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            },
            'wilcoxon_test': {
                'statistic': float(w_stat),
                'p_value': float(w_p_value),
                'significant': w_p_value < 0.05
            },
            'sample_size': len(baseline_similarities)
        }
    
    def _disable_component(self, component: str):
        """Temporarily disable a model component."""
        # This is a simplified implementation
        # In practice, you'd need component-specific disabling logic
        if hasattr(self.model, component):
            setattr(self.model, f'_{component}_enabled', False)
    
    def _enable_component(self, component: str):
        """Re-enable a model component."""
        if hasattr(self.model, component):
            setattr(self.model, f'_{component}_enabled', True)
    
    def _analyze_component_impact(self, ablation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the impact of each ablated component."""
        baseline = ablation_results.get('baseline', {})
        impact_analysis = {}
        
        for key, results in ablation_results.items():
            if key.startswith('without_') and isinstance(results, dict):
                component = key.replace('without_', '')
                
                # Compare key metrics
                impact = {}
                for metric in ['validity', 'uniqueness', 'novelty', 'scaffold_preservation_rate']:
                    baseline_val = baseline.get(metric, 0)
                    ablated_val = results.get(metric, 0)
                    
                    if baseline_val != 0:
                        impact[metric] = (baseline_val - ablated_val) / baseline_val * 100
                    else:
                        impact[metric] = 0.0
                
                impact_analysis[component] = impact
        
        return impact_analysis
    
    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to evaluation device."""
        def move_item(item):
            if isinstance(item, torch.Tensor):
                return item.to(self.device)
            elif isinstance(item, dict):
                return {k: move_item(v) for k, v in item.items()}
            elif hasattr(item, 'to'):
                return item.to(self.device)
            else:
                return item
        
        return {k: move_item(v) for k, v in batch.items()}
    
    def _save_evaluation_results(self,
                               results: Dict[str, Any],
                               predictions: List[str],
                               targets: List[str]):
        """Save detailed evaluation results."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save metrics
        metrics_file = self.output_dir / f'evaluation_metrics_{timestamp}.json'
        with open(metrics_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_results = self._make_json_serializable(results)
            json.dump(serializable_results, f, indent=2)
        
        # Save predictions and targets
        data_file = self.output_dir / f'evaluation_data_{timestamp}.csv'
        df = pd.DataFrame({
            'predictions': predictions,
            'targets': targets
        })
        df.to_csv(data_file, index=False)
        
        logger.info(f"Evaluation results saved to {self.output_dir}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to JSON serializable types."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


class BenchmarkEvaluator:
    """
    Standardized benchmark evaluation for molecular generation models.
    """
    
    def __init__(self, benchmark_name: str = 'scaffold_mol_benchmark'):
        """
        Initialize benchmark evaluator.
        
        Args:
            benchmark_name: Name of the benchmark
        """
        self.benchmark_name = benchmark_name
        self.benchmark_metrics = BenchmarkMetrics()
        
        # Standard benchmark configurations
        self.benchmark_configs = {
            'validity_test': {'focus': 'validity', 'samples': 1000},
            'diversity_test': {'focus': 'diversity', 'samples': 1000},
            'novelty_test': {'focus': 'novelty', 'samples': 1000},
            'scaffold_preservation_test': {'focus': 'scaffold', 'samples': 500}
        }
    
    def run_full_benchmark(self,
                          model: ScaffoldBasedMolT5Generator,
                          test_dataloader: DataLoader,
                          train_data: List[str]) -> Dict[str, Any]:
        """
        Run complete benchmark evaluation.
        
        Args:
            model: Model to evaluate
            test_dataloader: Test data loader
            train_data: Training data for novelty computation
            
        Returns:
            Complete benchmark results
        """
        logger.info(f"Running full benchmark: {self.benchmark_name}")
        
        evaluator = ModelEvaluator(model)
        benchmark_results = {}
        
        # Run individual benchmark tests
        for test_name, config in self.benchmark_configs.items():
            logger.info(f"Running {test_name}...")
            
            test_results = evaluator.evaluate_dataset(
                test_dataloader,
                num_samples=config['samples'],
                save_results=False
            )
            
            benchmark_results[test_name] = test_results
        
        # Aggregate results
        aggregated_results = self._aggregate_benchmark_results(benchmark_results)
        
        # Compute final benchmark score
        final_score = self._compute_benchmark_score(aggregated_results)
        
        results = {
            'benchmark_name': self.benchmark_name,
            'individual_tests': benchmark_results,
            'aggregated_results': aggregated_results,
            'final_score': final_score
        }
        
        return results
    
    def _aggregate_benchmark_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from individual benchmark tests."""
        # Extract key metrics from each test
        aggregated = {}
        
        metric_keys = ['validity', 'uniqueness', 'novelty', 'diversity_score', 
                      'scaffold_preservation_rate', 'lipinski_compliance']
        
        for metric in metric_keys:
            values = []
            for test_results in results.values():
                if metric in test_results:
                    values.append(test_results[metric])
            
            if values:
                aggregated[f'mean_{metric}'] = np.mean(values)
                aggregated[f'std_{metric}'] = np.std(values)
        
        return aggregated
    
    def _compute_benchmark_score(self, aggregated_results: Dict[str, Any]) -> float:
        """Compute overall benchmark score."""
        # Weighted combination of key metrics
        weights = {
            'mean_validity': 0.25,
            'mean_uniqueness': 0.20,
            'mean_novelty': 0.20,
            'mean_diversity_score': 0.15,
            'mean_scaffold_preservation_rate': 0.15,
            'mean_lipinski_compliance': 0.05
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in aggregated_results:
                score += aggregated_results[metric] * weight
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0