#!/usr/bin/env python3
"""
Comprehensive evaluation of Stage 2 model with all metrics including FCD.
Evaluates on real ChEBI-20 test dataset.
"""

import os
import sys
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project path
sys.path.append('/root/text2Mol/scaffold-mol-generation')

from scaffold_mol_gen.models.end2end_model import End2EndMolecularGenerator
from scaffold_mol_gen.evaluation.comprehensive_metrics import ComprehensiveMetrics
from rdkit import Chem

class SmilesValidator:
    """Simple SMILES validator using RDKit."""
    def validate_smiles(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return True, Chem.MolToSmiles(mol, canonical=True)
            return False, None
        except:
            return False, None

def load_model(checkpoint_path):
    """Load trained Stage 2 model."""
    print(f"Loading model from {checkpoint_path}")
    
    # Initialize model
    model = End2EndMolecularGenerator(
        hidden_size=768,
        molt5_path='/root/autodl-tmp/text2Mol-models/molt5-base',
        use_scibert=False,
        freeze_encoders=False,
        freeze_molt5=False,
        fusion_type='both',
        device='cuda'
    )
    
    # Load checkpoint (weights_only=False is safe since this is our own trained model)
    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        # Remove 'model.' prefix from keys if present
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v  # Remove 'model.' prefix
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    model.cuda()
    
    print("✅ Model loaded successfully")
    return model

def generate_molecules_batch(model, scaffold_batch, text_batch, scaffold_modality='smiles'):
    """Generate molecules for a batch of inputs."""
    try:
        with torch.no_grad():
            generated = model.generate(
                scaffold_batch,
                text_batch,
                scaffold_modality=scaffold_modality,
                max_length=256,
                num_beams=5,
                temperature=0.8
            )
        return generated
    except Exception as e:
        print(f"Generation error: {e}")
        return [''] * len(scaffold_batch)

def evaluate_model(model_path, test_data_path, output_dir, num_samples=None):
    """Main evaluation function."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = load_model(model_path)
    
    # Load test data
    print(f"\nLoading test data from {test_data_path}")
    test_df = pd.read_csv(test_data_path)
    
    # Use subset if specified
    if num_samples:
        test_df = test_df.head(num_samples)
        print(f"Using {num_samples} samples for evaluation")
    else:
        print(f"Using all {len(test_df)} test samples")
    
    # Ensure required columns exist and map them correctly
    # Map 'description' to 'text' if needed
    if 'description' in test_df.columns and 'text' not in test_df.columns:
        test_df['text'] = test_df['description']
    
    required_cols = ['scaffold', 'text', 'SMILES']
    for col in required_cols:
        if col not in test_df.columns:
            print(f"Error: Required column '{col}' not found in test data")
            print(f"Available columns: {list(test_df.columns)}")
            return
    
    # Initialize metrics calculator
    metrics_calc = ComprehensiveMetrics()
    validator = SmilesValidator()
    
    # Generate molecules
    print("\n🔄 Generating molecules...")
    generated_smiles = []
    target_smiles = []
    valid_generated = []
    
    batch_size = 8
    num_batches = (len(test_df) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Generating"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(test_df))
        
        batch_df = test_df.iloc[start_idx:end_idx]
        
        # Prepare batch
        scaffold_batch = batch_df['scaffold'].tolist()
        text_batch = batch_df['text'].tolist()
        target_batch = batch_df['SMILES'].tolist()
        
        # Generate
        generated_batch = generate_molecules_batch(
            model, scaffold_batch, text_batch, scaffold_modality='smiles'
        )
        
        # Collect results
        for gen, tgt in zip(generated_batch, target_batch):
            generated_smiles.append(gen)
            target_smiles.append(tgt)
            
            # Validate generated SMILES
            is_valid, _ = validator.validate_smiles(gen)
            if is_valid:
                valid_generated.append(gen)
    
    print(f"\n✅ Generated {len(generated_smiles)} molecules")
    print(f"   Valid: {len(valid_generated)} ({100*len(valid_generated)/len(generated_smiles):.1f}%)")
    
    # Calculate all metrics
    print("\n📊 Calculating comprehensive metrics...")
    
    # Get training data for novelty and FCD calculation
    train_df = pd.read_csv('/root/text2Mol/scaffold-mol-generation/Datasets/train.csv')
    train_smiles = train_df['SMILES'].tolist()
    
    # Calculate metrics
    metrics_results = metrics_calc.calculate_all_metrics(
        generated=generated_smiles,
        targets=target_smiles,
        reference=train_smiles
    )
    
    # Format and display results
    print("\n" + "="*60)
    print("📈 EVALUATION RESULTS")
    print("="*60)
    
    # Core generation metrics
    print("\n🎯 Core Generation Metrics:")
    print(f"  Validity:    {metrics_results['validity']:.2%}")
    print(f"  Uniqueness:  {metrics_results['uniqueness']:.2%}")
    print(f"  Novelty:     {metrics_results['novelty']:.2%}")
    
    # Sequence similarity metrics
    print("\n📝 Sequence Similarity Metrics:")
    print(f"  BLEU-1:      {metrics_results['bleu1']:.4f}")
    print(f"  BLEU-2:      {metrics_results['bleu2']:.4f}")
    print(f"  BLEU-3:      {metrics_results['bleu3']:.4f}")
    print(f"  BLEU-4:      {metrics_results['bleu4']:.4f}")
    print(f"  Exact Match: {metrics_results['exact_match']:.2%}")
    print(f"  Levenshtein: {metrics_results['levenshtein']:.2f}")
    
    # Molecular fingerprint metrics
    print("\n🧬 Molecular Fingerprint Similarity:")
    print(f"  MACCS FTS:   {metrics_results['maccs_fts']:.4f}")
    print(f"  Morgan FTS:  {metrics_results['morgan_fts']:.4f}")
    print(f"  RDKit FTS:   {metrics_results['rdk_fts']:.4f}")
    
    # Distribution metric
    print("\n📊 Distribution Metric:")
    print(f"  FCD Score:   {metrics_results['fcd']:.4f}")
    
    # Summary statistics
    print("\n📋 Summary Statistics:")
    avg_fingerprint = np.mean([
        metrics_results['maccs_fts'],
        metrics_results['morgan_fts'],
        metrics_results['rdk_fts']
    ])
    print(f"  Average Fingerprint Similarity: {avg_fingerprint:.4f}")
    print(f"  Overall BLEU (avg 1-4):         {np.mean([metrics_results[f'bleu{i}'] for i in range(1,5)]):.4f}")
    
    # Save results to JSON
    results_file = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(metrics_results, f, indent=2)
    print(f"\n💾 Results saved to {results_file}")
    
    # Save generated molecules
    output_df = pd.DataFrame({
        'scaffold': test_df['scaffold'].iloc[:len(generated_smiles)],
        'text': test_df['text'].iloc[:len(generated_smiles)],
        'target_smiles': target_smiles,
        'generated_smiles': generated_smiles,
        'is_valid': [validator.validate_smiles(s)[0] for s in generated_smiles]
    })
    
    output_file = os.path.join(output_dir, 'generated_molecules.csv')
    output_df.to_csv(output_file, index=False)
    print(f"💾 Generated molecules saved to {output_file}")
    
    # Generate detailed report
    report_file = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("STAGE 2 MODEL EVALUATION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Model: {model_path}\n")
        f.write(f"Test Data: {test_data_path}\n")
        f.write(f"Samples Evaluated: {len(test_df)}\n\n")
        
        f.write("COMPREHENSIVE METRICS:\n")
        f.write("-"*40 + "\n")
        for metric, value in metrics_results.items():
            if isinstance(value, float):
                if metric in ['validity', 'uniqueness', 'novelty', 'exact_match']:
                    f.write(f"{metric:20s}: {value:8.2%}\n")
                else:
                    f.write(f"{metric:20s}: {value:8.4f}\n")
            else:
                f.write(f"{metric:20s}: {value}\n")
    
    print(f"📄 Detailed report saved to {report_file}")
    
    return metrics_results

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "/root/autodl-tmp/text2Mol-stage2/best_model_stage2.pt"
    TEST_DATA = "/root/text2Mol/scaffold-mol-generation/Datasets/test_with_scaffold.csv"
    OUTPUT_DIR = f"/root/autodl-tmp/text2Mol-evaluation-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Run evaluation (use subset for faster testing, set to None for full evaluation)
    print("🚀 Starting comprehensive evaluation...")
    print("   This will evaluate all metrics including FCD")
    print("   Using real ChEBI-20 test dataset\n")
    
    # Use 100 samples for faster evaluation, set to None for full test set
    results = evaluate_model(MODEL_PATH, TEST_DATA, OUTPUT_DIR, num_samples=100)
    
    print("\n✅ Evaluation complete!")
    print(f"   Check {OUTPUT_DIR} for detailed results")