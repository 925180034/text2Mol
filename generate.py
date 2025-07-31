#!/usr/bin/env python3
"""
Generation script for scaffold-based molecular generation.

This script provides interactive molecule generation capabilities including:
- Single molecule generation
- Batch generation from CSV
- Interactive generation with user input
- Quality assessment of generated molecules
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import json
import pandas as pd
import torch
from transformers import T5Tokenizer

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from scaffold_mol_gen.models.core_model import ScaffoldBasedMolT5Generator
from scaffold_mol_gen.utils.mol_utils import MolecularUtils
from scaffold_mol_gen.utils.scaffold_utils import ScaffoldExtractor
from scaffold_mol_gen.utils.visualization import MolecularVisualizer
from scaffold_mol_gen.training.metrics import evaluate_model_outputs

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate molecules using scaffold-based model')
    
    parser.add_argument(
        '--model-checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint file'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to model configuration file'
    )
    
    # Generation modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    
    mode_group.add_argument(
        '--interactive',
        action='store_true',
        help='Interactive generation mode'
    )
    
    mode_group.add_argument(
        '--single',
        action='store_true',
        help='Single molecule generation'
    )
    
    mode_group.add_argument(
        '--batch',
        type=str,
        help='Batch generation from CSV file'
    )
    
    # Single generation parameters
    parser.add_argument(
        '--scaffold',
        type=str,
        help='Input scaffold SMILES (for single generation)'
    )
    
    parser.add_argument(
        '--text',
        type=str,
        help='Input text description (for single generation)'
    )
    
    # Generation parameters
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Number of samples to generate per input'
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=200,
        help='Maximum generation length'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Generation temperature'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=50,
        help='Top-k sampling parameter'
    )
    
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.95,
        help='Top-p (nucleus) sampling parameter'
    )
    
    parser.add_argument(
        '--num-beams',
        type=int,
        default=5,
        help='Number of beams for beam search'
    )
    
    parser.add_argument(
        '--do-sample',
        action='store_true',
        default=True,
        help='Use sampling instead of greedy decoding'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='generation_results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save generation results to files'
    )
    
    parser.add_argument(
        '--create-visualizations',
        action='store_true',
        help='Create molecular visualizations'
    )
    
    parser.add_argument(
        '--assess-quality',
        action='store_true',
        help='Assess quality of generated molecules'
    )
    
    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use for generation (auto, cpu, cuda)'
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
    """Setup generation device."""
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
        model = ScaffoldBasedMolT5Generator.from_pretrained(checkpoint_path, config['model'])
    except:
        logger.info("Using fallback loading method...")
        model = ScaffoldBasedMolT5Generator(config['model'])
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    model_info = model.get_model_info()
    logger.info(f"Loaded model: {model_info['model_name']} v{model_info['version']}")
    
    return model

def generate_molecules(model: ScaffoldBasedMolT5Generator,
                      scaffold: str,
                      text: str,
                      generation_config: dict) -> list:
    """Generate molecules using the model."""
    logger.info(f"Generating molecules for scaffold: {scaffold}")
    logger.info(f"Text description: {text}")
    
    try:
        generated_molecules = model.generate(
            scaffold=scaffold,
            text=text,
            **generation_config
        )
        
        logger.info(f"Generated {len(generated_molecules)} molecules")
        return generated_molecules
        
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        return []

def assess_molecule_quality(molecules: list, scaffold: str) -> dict:
    """Assess quality of generated molecules."""
    logger.info("Assessing molecule quality...")
    
    quality_metrics = {}
    
    # Basic validity
    valid_molecules = [mol for mol in molecules if MolecularUtils.validate_smiles(mol)]
    quality_metrics['validity'] = len(valid_molecules) / len(molecules) if molecules else 0
    quality_metrics['valid_count'] = len(valid_molecules)
    quality_metrics['total_count'] = len(molecules)
    
    if not valid_molecules:
        return quality_metrics
    
    # Uniqueness
    unique_molecules = set(MolecularUtils.canonicalize_smiles(mol) for mol in valid_molecules)
    unique_molecules = {mol for mol in unique_molecules if mol}
    quality_metrics['uniqueness'] = len(unique_molecules) / len(valid_molecules)
    quality_metrics['unique_count'] = len(unique_molecules)
    
    # Scaffold preservation
    scaffold_extractor = ScaffoldExtractor()
    preserved_count = 0
    
    for mol in valid_molecules:
        mol_scaffold = scaffold_extractor.get_murcko_scaffold(mol)
        if mol_scaffold == scaffold:
            preserved_count += 1
    
    quality_metrics['scaffold_preservation'] = preserved_count / len(valid_molecules)
    quality_metrics['scaffold_preserved_count'] = preserved_count
    
    # Drug-likeness (Lipinski's Rule of Five)
    drug_like_count = 0
    for mol in valid_molecules:
        props = MolecularUtils.compute_molecular_properties(mol)
        if props:
            mw = props.get('molecular_weight', 0)
            logp = props.get('logp', 0)
            hbd = props.get('num_hbd', 0)
            hba = props.get('num_hba', 0)
            
            if mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10:
                drug_like_count += 1
    
    quality_metrics['drug_likeness'] = drug_like_count / len(valid_molecules)
    quality_metrics['drug_like_count'] = drug_like_count
    
    return quality_metrics

def create_visualizations(molecules: list, scaffold: str, output_dir: Path):
    """Create visualizations of generated molecules."""
    logger.info("Creating visualizations...")
    
    visualizer = MolecularVisualizer()
    
    # Filter valid molecules
    valid_molecules = [mol for mol in molecules if MolecularUtils.validate_smiles(mol)]
    
    if not valid_molecules:
        logger.warning("No valid molecules to visualize")
        return
    
    try:
        # Create molecule grid
        grid_image = visualizer.create_molecule_grid(
            valid_molecules[:20],  # Show up to 20 molecules
            titles=[f"Mol {i+1}" for i in range(min(20, len(valid_molecules)))]
        )
        grid_image.save(output_dir / 'generated_molecules_grid.png')
        
        # Plot molecular properties
        fig = visualizer.plot_molecular_properties(valid_molecules)
        visualizer.save_visualization(fig, output_dir / 'molecular_properties.png')
        
        logger.info(f"Visualizations saved to {output_dir}")
        
    except Exception as e:
        logger.warning(f"Error creating visualizations: {e}")

def interactive_generation(model: ScaffoldBasedMolT5Generator, 
                         generation_config: dict,
                         output_dir: Path):
    """Interactive generation mode."""
    logger.info("Starting interactive generation mode...")
    logger.info("Enter 'quit' or 'exit' to stop")
    
    while True:
        try:
            # Get user input
            print("\n" + "="*50)
            scaffold = input("Enter scaffold SMILES: ").strip()
            
            if scaffold.lower() in ['quit', 'exit']:
                break
            
            if not MolecularUtils.validate_smiles(scaffold):
                print("Invalid scaffold SMILES. Please try again.")
                continue
            
            text = input("Enter text description: ").strip()
            
            if not text:
                print("Text description cannot be empty. Please try again.")
                continue
            
            # Generate molecules
            print(f"\nGenerating {generation_config['num_samples']} molecules...")
            molecules = generate_molecules(model, scaffold, text, generation_config)
            
            if not molecules:
                print("No molecules generated. Please try again.")
                continue
            
            # Display results
            print(f"\nGenerated {len(molecules)} molecules:")
            for i, mol in enumerate(molecules, 1):
                valid = "✓" if MolecularUtils.validate_smiles(mol) else "✗"
                print(f"{i:2d}. {mol} {valid}")
            
            # Assess quality
            quality = assess_molecule_quality(molecules, scaffold)
            print(f"\nQuality Assessment:")
            print(f"  Validity: {quality['validity']:.3f} ({quality['valid_count']}/{quality['total_count']})")
            print(f"  Uniqueness: {quality.get('uniqueness', 0):.3f}")
            print(f"  Scaffold Preservation: {quality.get('scaffold_preservation', 0):.3f}")
            print(f"  Drug-likeness: {quality.get('drug_likeness', 0):.3f}")
            
            # Ask if user wants to save results
            save = input("\nSave results? (y/n): ").strip().lower()
            if save in ['y', 'yes']:
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                results_file = output_dir / f'interactive_results_{timestamp}.json'
                
                results = {
                    'scaffold': scaffold,
                    'text': text,
                    'generated_molecules': molecules,
                    'quality_metrics': quality,
                    'generation_config': generation_config
                }
                
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                print(f"Results saved to: {results_file}")
        
        except KeyboardInterrupt:
            print("\nExiting interactive mode...")
            break
        
        except Exception as e:
            print(f"Error: {e}")
            continue

def single_generation(model: ScaffoldBasedMolT5Generator,
                     scaffold: str,
                     text: str,
                     generation_config: dict,
                     output_dir: Path,
                     args):
    """Single molecule generation."""
    logger.info("Running single generation...")
    
    # Validate inputs
    if not MolecularUtils.validate_smiles(scaffold):
        logger.error(f"Invalid scaffold SMILES: {scaffold}")
        return
    
    if not text:
        logger.error("Text description cannot be empty")
        return
    
    # Generate molecules
    molecules = generate_molecules(model, scaffold, text, generation_config)
    
    if not molecules:
        logger.error("No molecules generated")
        return
    
    # Display results
    print(f"\nGenerated {len(molecules)} molecules:")
    for i, mol in enumerate(molecules, 1):
        valid = "✓" if MolecularUtils.validate_smiles(mol) else "✗"
        print(f"{i:2d}. {mol} {valid}")
    
    # Assess quality
    if args.assess_quality:
        quality = assess_molecule_quality(molecules, scaffold)
        print(f"\nQuality Assessment:")
        for metric, value in quality.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value}")
    
    # Create visualizations
    if args.create_visualizations:
        create_visualizations(molecules, scaffold, output_dir)
    
    # Save results
    if args.save_results:
        results = {
            'scaffold': scaffold,
            'text': text,
            'generated_molecules': molecules,
            'generation_config': generation_config
        }
        
        if args.assess_quality:
            results['quality_metrics'] = assess_molecule_quality(molecules, scaffold)
        
        results_file = output_dir / 'single_generation_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")

def batch_generation(model: ScaffoldBasedMolT5Generator,
                    batch_file: str,
                    generation_config: dict,
                    output_dir: Path,
                    args):
    """Batch generation from CSV file."""
    logger.info(f"Running batch generation from: {batch_file}")
    
    # Load batch data
    try:
        df = pd.read_csv(batch_file)
        logger.info(f"Loaded {len(df)} entries for batch generation")
    except Exception as e:
        logger.error(f"Error loading batch file: {e}")
        return
    
    # Validate required columns
    required_columns = ['scaffold', 'text']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return
    
    # Process each entry
    all_results = []
    
    for idx, row in df.iterrows():
        scaffold = row['scaffold']
        text = row['text']
        
        logger.info(f"Processing entry {idx+1}/{len(df)}: {scaffold}")
        
        # Validate scaffold
        if not MolecularUtils.validate_smiles(scaffold):
            logger.warning(f"Invalid scaffold at row {idx}: {scaffold}")
            continue
        
        # Generate molecules
        molecules = generate_molecules(model, scaffold, text, generation_config)
        
        # Assess quality
        quality = assess_molecule_quality(molecules, scaffold) if args.assess_quality else {}
        
        result = {
            'index': idx,
            'scaffold': scaffold,
            'text': text,
            'generated_molecules': molecules,
            'quality_metrics': quality
        }
        
        all_results.append(result)
    
    # Save batch results
    if args.save_results:
        results_file = output_dir / 'batch_generation_results.json'
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Also save as CSV for easy analysis
        csv_data = []
        for result in all_results:
            for i, mol in enumerate(result['generated_molecules']):
                csv_data.append({
                    'input_index': result['index'],
                    'scaffold': result['scaffold'],
                    'text': result['text'],
                    'generated_molecule': mol,
                    'sample_index': i,
                    'valid': MolecularUtils.validate_smiles(mol)
                })
        
        csv_df = pd.DataFrame(csv_data)
        csv_df.to_csv(output_dir / 'batch_generation_results.csv', index=False)
        
        logger.info(f"Batch results saved to: {output_dir}")
    
    # Print summary
    total_generated = sum(len(r['generated_molecules']) for r in all_results)
    total_valid = sum(sum(1 for mol in r['generated_molecules'] 
                         if MolecularUtils.validate_smiles(mol)) for r in all_results)
    
    print(f"\nBatch Generation Summary:")
    print(f"  Processed entries: {len(all_results)}")
    print(f"  Total molecules generated: {total_generated}")
    print(f"  Total valid molecules: {total_valid}")
    print(f"  Overall validity: {total_valid/total_generated:.3f}" if total_generated > 0 else "  Overall validity: 0.000")

def main():
    """Main generation function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = setup_device(args.device)
    
    logger.info("=" * 50)
    logger.info("SCAFFOLD-BASED MOLECULAR GENERATION")
    logger.info("=" * 50)
    
    # Load model
    model = load_model(args.model_checkpoint, config, device)
    
    # Setup generation configuration
    generation_config = {
        'num_samples': args.num_samples,
        'max_length': args.max_length,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'num_beams': args.num_beams,
        'do_sample': args.do_sample
    }
    
    logger.info(f"Generation configuration: {generation_config}")
    
    # Run appropriate generation mode
    if args.interactive:
        interactive_generation(model, generation_config, output_dir)
    
    elif args.single:
        if not args.scaffold or not args.text:
            logger.error("Both --scaffold and --text are required for single generation")
            return
        
        single_generation(model, args.scaffold, args.text, generation_config, output_dir, args)
    
    elif args.batch:
        batch_generation(model, args.batch, generation_config, output_dir, args)
    
    logger.info("Generation script completed!")

if __name__ == '__main__':
    main()