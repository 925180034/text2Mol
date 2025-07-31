#!/usr/bin/env python3
"""
Evaluation script for scaffold-based molecular generation.

This script provides comprehensive evaluation of trained models including:
- Performance metrics computation
- Benchmark evaluation
- Comparative analysis
- Visualization generation
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import json
import torch
import pandas as pd
from transformers import T5Tokenizer

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from scaffold_mol_gen.models.core_model import ScaffoldBasedMolT5Generator
from scaffold_mol_gen.data.dataset import ScaffoldMolDataset, MultiModalMolDataset
from scaffold_mol_gen.data.collate import create_data_loader
from scaffold_mol_gen.evaluation.evaluator import ModelEvaluator, BenchmarkEvaluator
from scaffold_mol_gen.training.metrics import evaluate_model_outputs
from scaffold_mol_gen.utils.visualization import MolecularVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate scaffold-based molecular generation model')
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/evaluation_config.yaml',
        help='Path to evaluation configuration file'
    )
    
    parser.add_argument(
        '--model-checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint file'
    )
    
    parser.add_argument(
        '--test-data',
        type=str,
        help='Path to test data CSV file (overrides config)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation_results',
        help='Output directory for evaluation results'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        help='Number of samples to evaluate (overrides config)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    
    parser.add_argument(
        '--generation-samples',
        type=int,
        default=1,
        help='Number of samples to generate per input'
    )
    
    parser.add_argument(
        '--run-benchmark',
        action='store_true',
        help='Run standardized benchmark evaluation'
    )
    
    parser.add_argument(
        '--create-visualizations',
        action='store_true',
        help='Create visualization plots'
    )
    
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save model predictions to file'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use for evaluation (auto, cpu, cuda)'
    )
    
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise

def setup_device(device_arg: str) -> torch.device:
    """Setup evaluation device."""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    logger.info(f"Using device: {device}")
    return device

def load_model(checkpoint_path: str, config: dict, device: torch.device) -> ScaffoldBasedMolT5Generator:
    """Load model from checkpoint."""
    logger.info(f"Loading model from: {checkpoint_path}")
    
    try:
        # Try to load using the model's from_pretrained method
        model = ScaffoldBasedMolT5Generator.from_pretrained(checkpoint_path, config['model'])
    except:
        # Fallback: create model and load state dict
        logger.info("Using fallback loading method...")
        model = ScaffoldBasedMolT5Generator(config['model'])
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Print model info
    model_info = model.get_model_info()
    logger.info(f"Loaded model: {model_info['model_name']} v{model_info['version']}")
    logger.info(f"Total parameters: {model_info['total_parameters']:,}")
    
    return model

def create_evaluation_dataset(config: dict, tokenizer, data_path: str):
    """Create evaluation dataset."""
    data_config = config['data']
    
    logger.info(f"Loading evaluation data from: {data_path}")
    
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
    
    # Create dataset
    if data_config.get('combination_type'):
        dataset = MultiModalMolDataset(
            data_path=data_path,
            combination_type=data_config['combination_type'],
            **dataset_params
        )
    else:
        dataset = ScaffoldMolDataset(
            data_path=data_path,
            **dataset_params
        )
    
    logger.info(f"Evaluation dataset size: {len(dataset)}")
    return dataset

def run_model_evaluation(evaluator: ModelEvaluator, 
                        dataloader, 
                        config: dict, 
                        args) -> dict:
    """Run comprehensive model evaluation."""
    logger.info("Starting model evaluation...")
    
    # Generation configuration
    eval_config = config.get('evaluation', {})
    generation_config = eval_config.get('generation', {})
    generation_config['num_samples'] = args.generation_samples
    
    # Run evaluation
    results = evaluator.evaluate_dataset(
        dataloader=dataloader,
        num_samples=args.num_samples,
        generation_config=generation_config,
        save_results=True
    )
    
    return results

def run_benchmark_evaluation(model: ScaffoldBasedMolT5Generator,
                           dataloader,
                           config: dict) -> dict:
    """Run standardized benchmark evaluation."""
    logger.info("Running benchmark evaluation...")
    
    benchmark_evaluator = BenchmarkEvaluator("scaffold_mol_benchmark")
    
    # Create training data reference (for novelty computation)
    # In practice, you'd load the actual training data
    train_data = []  # Placeholder
    
    results = benchmark_evaluator.run_full_benchmark(
        model=model,
        test_dataloader=dataloader,
        train_data=train_data
    )
    
    return results

def create_visualizations(results: dict, predictions: list, targets: list, output_dir: Path):
    """Create evaluation visualizations."""
    logger.info("Creating visualizations...")
    
    visualizer = MolecularVisualizer()
    
    try:
        # Molecular property distributions
        if predictions:
            fig = visualizer.plot_molecular_properties(predictions)
            visualizer.save_visualization(fig, output_dir / 'molecular_properties.png')
        
        # Generation results comparison
        if predictions and targets:
            scaffolds = ['' for _ in predictions]  # Placeholder
            fig = visualizer.plot_generation_results(targets, predictions, scaffolds)
            visualizer.save_visualization(fig, output_dir / 'generation_results.png')
        
        # Molecule grid visualization
        if predictions:
            sample_molecules = predictions[:20]  # Show first 20
            grid_image = visualizer.create_molecule_grid(sample_molecules)
            grid_image.save(output_dir / 'molecule_grid.png')
        
        logger.info(f"Visualizations saved to {output_dir}")
        
    except Exception as e:
        logger.warning(f"Error creating visualizations: {e}")

def save_evaluation_results(results: dict, 
                          predictions: list, 
                          targets: list,
                          output_dir: Path,
                          save_predictions: bool):
    """Save evaluation results to files."""
    logger.info("Saving evaluation results...")
    
    # Save metrics as JSON
    metrics_file = output_dir / 'evaluation_metrics.json'
    with open(metrics_file, 'w') as f:
        # Convert numpy types for JSON serialization
        serializable_results = convert_numpy_types(results)
        json.dump(serializable_results, f, indent=2)
    
    # Save detailed results as CSV if predictions available
    if save_predictions and predictions and targets:
        results_df = pd.DataFrame({
            'predictions': predictions,
            'targets': targets,
            'valid_prediction': [is_valid_smiles(p) for p in predictions],
            'valid_target': [is_valid_smiles(t) for t in targets]
        })
        
        results_df.to_csv(output_dir / 'detailed_results.csv', index=False)
    
    # Create summary report
    create_summary_report(results, output_dir)
    
    logger.info(f"Results saved to {output_dir}")

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    import numpy as np
    
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def is_valid_smiles(smiles: str) -> bool:
    """Check if SMILES string is valid."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

def create_summary_report(results: dict, output_dir: Path):
    """Create a human-readable summary report."""
    report_path = output_dir / 'evaluation_summary.txt'
    
    with open(report_path, 'w') as f:
        f.write("SCAFFOLD-BASED MOLECULAR GENERATION EVALUATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        # Key metrics
        key_metrics = [
            'validity', 'uniqueness', 'novelty', 'diversity_score',
            'scaffold_preservation_rate', 'lipinski_compliance'
        ]
        
        f.write("KEY METRICS:\n")
        f.write("-" * 20 + "\n")
        
        for metric in key_metrics:
            if metric in results:
                value = results[metric]
                if isinstance(value, float):
                    f.write(f"{metric:25s}: {value:.4f}\n")
                else:
                    f.write(f"{metric:25s}: {value}\n")
        
        f.write("\n")
        
        # Evaluation metadata
        if 'evaluation_metadata' in results:
            metadata = results['evaluation_metadata']
            f.write("EVALUATION DETAILS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Samples evaluated: {metadata.get('num_samples_evaluated', 'N/A')}\n")
            f.write(f"Evaluation time: {metadata.get('evaluation_time_seconds', 0):.2f}s\n")
            f.write(f"Device: {metadata.get('device', 'N/A')}\n")
        
        f.write("\n")
        
        # Additional metrics
        f.write("ADDITIONAL METRICS:\n")
        f.write("-" * 20 + "\n")
        
        for key, value in results.items():
            if key not in key_metrics and key != 'evaluation_metadata':
                if isinstance(value, (int, float)):
                    if isinstance(value, float):
                        f.write(f"{key:25s}: {value:.4f}\n")
                    else:
                        f.write(f"{key:25s}: {value}\n")

def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = setup_device(args.device)
    
    logger.info("=" * 50)
    logger.info("SCAFFOLD-BASED MOLECULAR GENERATION EVALUATION")
    logger.info("=" * 50)
    
    # Initialize tokenizer
    molt5_checkpoint = config['model']['molt5_checkpoint']
    logger.info(f"Loading tokenizer from: {molt5_checkpoint}")
    tokenizer = T5Tokenizer.from_pretrained(molt5_checkpoint)
    
    # Load model
    model = load_model(args.model_checkpoint, config, device)
    
    # Create evaluation dataset
    test_data_path = args.test_data or config['evaluation']['datasets']['test']
    dataset = create_evaluation_dataset(config, tokenizer, test_data_path)
    
    # Create data loader
    dataloader = create_data_loader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.get('infrastructure', {}).get('num_workers', 4)
    )
    
    # Create evaluator
    evaluator = ModelEvaluator(model, device, str(output_dir))
    
    # Run evaluation
    results = run_model_evaluation(evaluator, dataloader, config, args)
    
    # Extract predictions and targets for additional processing
    predictions = []
    targets = []
    
    # Run benchmark evaluation if requested
    if args.run_benchmark:
        benchmark_results = run_benchmark_evaluation(model, dataloader, config)
        results['benchmark_results'] = benchmark_results
    
    # Create visualizations if requested
    if args.create_visualizations:
        create_visualizations(results, predictions, targets, output_dir)
    
    # Save results
    save_evaluation_results(results, predictions, targets, output_dir, args.save_predictions)
    
    # Print summary
    logger.info("\nEVALUATION COMPLETED!")
    logger.info(f"Results saved to: {output_dir}")
    
    # Print key metrics
    key_metrics = ['validity', 'uniqueness', 'novelty', 'scaffold_preservation_rate']
    logger.info("\nKEY METRICS:")
    for metric in key_metrics:
        if metric in results:
            logger.info(f"  {metric}: {results[metric]:.4f}")
    
    logger.info("Evaluation script completed!")

if __name__ == '__main__':
    main()