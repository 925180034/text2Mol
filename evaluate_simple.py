#!/usr/bin/env python3
"""
Simple evaluation to test current training results with all available metrics.
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit import DataStructs
import warnings
warnings.filterwarnings('ignore')

# Add project path
sys.path.append('/root/text2Mol/scaffold-mol-generation')

from scaffold_mol_gen.models.end2end_model import End2EndMolecularGenerator
from scaffold_mol_gen.evaluation.comprehensive_metrics import ComprehensiveMetrics

def load_model(checkpoint_path):
    """Load trained Stage 2 model."""
    print(f"Loading model from {checkpoint_path}")
    
    model = End2EndMolecularGenerator(
        hidden_size=768,
        molt5_path='/root/autodl-tmp/text2Mol-models/molt5-base',
        use_scibert=False,
        freeze_encoders=False,
        freeze_molt5=False,
        fusion_type='both',
        device='cuda'
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
    
    model.eval()
    model.cuda()
    print("✅ Model loaded successfully")
    return model

def evaluate_metrics(generated_smiles, target_smiles, train_smiles):
    """Calculate all available metrics."""
    
    results = {}
    
    # 1. Validity
    valid_smiles = []
    for smi in generated_smiles:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_smiles.append(Chem.MolToSmiles(mol, canonical=True))
        except:
            pass
    
    results['validity'] = len(valid_smiles) / len(generated_smiles) if generated_smiles else 0
    results['num_valid'] = len(valid_smiles)
    results['num_total'] = len(generated_smiles)
    
    # 2. Uniqueness
    if valid_smiles:
        unique_smiles = set(valid_smiles)
        results['uniqueness'] = len(unique_smiles) / len(valid_smiles)
        results['num_unique'] = len(unique_smiles)
    else:
        results['uniqueness'] = 0
        results['num_unique'] = 0
    
    # 3. Novelty
    if valid_smiles and train_smiles:
        train_set = set(train_smiles)
        novel_smiles = [s for s in valid_smiles if s not in train_set]
        results['novelty'] = len(novel_smiles) / len(valid_smiles) if valid_smiles else 0
        results['num_novel'] = len(novel_smiles)
    else:
        results['novelty'] = 0
        results['num_novel'] = 0
    
    # 4. Exact Match
    exact_matches = sum(1 for gen, tgt in zip(generated_smiles, target_smiles) if gen == tgt)
    results['exact_match'] = exact_matches / len(generated_smiles) if generated_smiles else 0
    results['num_exact_match'] = exact_matches
    
    # 5. Fingerprint Similarity (only for valid molecules)
    if valid_smiles and len(valid_smiles) > 0:
        similarities = []
        for gen_smi, tgt_smi in zip(generated_smiles[:len(target_smiles)], target_smiles):
            try:
                gen_mol = Chem.MolFromSmiles(gen_smi)
                tgt_mol = Chem.MolFromSmiles(tgt_smi)
                if gen_mol and tgt_mol:
                    # Morgan fingerprint
                    gen_fp = AllChem.GetMorganFingerprintAsBitVect(gen_mol, 2, nBits=2048)
                    tgt_fp = AllChem.GetMorganFingerprintAsBitVect(tgt_mol, 2, nBits=2048)
                    sim = DataStructs.TanimotoSimilarity(gen_fp, tgt_fp)
                    similarities.append(sim)
            except:
                pass
        
        results['avg_similarity'] = np.mean(similarities) if similarities else 0
        results['num_compared'] = len(similarities)
    else:
        results['avg_similarity'] = 0
        results['num_compared'] = 0
    
    # 6. Try to calculate FCD if possible
    try:
        metrics_calc = ComprehensiveMetrics()
        full_metrics = metrics_calc.calculate_all_metrics(
            generated=generated_smiles,
            targets=target_smiles,
            reference=train_smiles
        )
        
        # Add any additional metrics that were calculated
        for key in ['fcd', 'maccs_fts', 'morgan_fts', 'rdk_fts']:
            if key in full_metrics:
                results[key] = full_metrics[key]
    except Exception as e:
        print(f"Note: Some advanced metrics could not be calculated: {e}")
    
    return results

def main():
    # Configuration
    MODEL_PATH = "/root/autodl-tmp/text2Mol-stage2/best_model_stage2.pt"
    TEST_DATA = "/root/text2Mol/scaffold-mol-generation/Datasets/test_with_scaffold.csv"
    TRAIN_DATA = "/root/text2Mol/scaffold-mol-generation/Datasets/train.csv"
    NUM_SAMPLES = 50  # Small sample for quick evaluation
    
    print("="*60)
    print("🧪 STAGE 2 MODEL EVALUATION - COMPREHENSIVE METRICS")
    print("="*60)
    
    # Load model
    model = load_model(MODEL_PATH)
    
    # Load data
    print(f"\n📂 Loading test data...")
    test_df = pd.read_csv(TEST_DATA).head(NUM_SAMPLES)
    train_df = pd.read_csv(TRAIN_DATA)
    
    if 'description' in test_df.columns and 'text' not in test_df.columns:
        test_df['text'] = test_df['description']
    
    print(f"   Using {len(test_df)} test samples")
    
    # Generate molecules
    print(f"\n🔬 Generating molecules...")
    generated_smiles = []
    target_smiles = test_df['SMILES'].tolist()
    train_smiles = train_df['SMILES'].tolist()
    
    batch_size = 8
    with torch.no_grad():
        for i in range(0, len(test_df), batch_size):
            batch_df = test_df.iloc[i:min(i+batch_size, len(test_df))]
            
            try:
                generated = model.generate(
                    batch_df['scaffold'].tolist(),
                    batch_df['text'].tolist(),
                    scaffold_modality='smiles',
                    max_length=256,
                    num_beams=5
                )
                generated_smiles.extend(generated)
            except Exception as e:
                print(f"   Generation error: {e}")
                generated_smiles.extend([''] * len(batch_df))
            
            print(f"   Progress: {min(i+batch_size, len(test_df))}/{len(test_df)}", end='\r')
    
    print(f"\n   Generated {len(generated_smiles)} SMILES")
    
    # Calculate metrics
    print(f"\n📊 Calculating evaluation metrics...")
    metrics = evaluate_metrics(generated_smiles, target_smiles, train_smiles)
    
    # Display results
    print("\n" + "="*60)
    print("📈 EVALUATION RESULTS")
    print("="*60)
    
    print(f"\n🎯 Core Metrics:")
    print(f"   Validity:      {metrics['validity']:.1%} ({metrics['num_valid']}/{metrics['num_total']})")
    print(f"   Uniqueness:    {metrics['uniqueness']:.1%} ({metrics['num_unique']}/{metrics['num_valid']})")
    print(f"   Novelty:       {metrics['novelty']:.1%} ({metrics['num_novel']}/{metrics['num_valid']})")
    print(f"   Exact Match:   {metrics['exact_match']:.1%} ({metrics['num_exact_match']}/{metrics['num_total']})")
    
    print(f"\n🧬 Similarity Metrics:")
    print(f"   Avg Tanimoto:  {metrics['avg_similarity']:.4f} (on {metrics['num_compared']} pairs)")
    
    if 'fcd' in metrics:
        print(f"\n📊 Advanced Metrics:")
        print(f"   FCD Score:     {metrics['fcd']:.4f}")
    if 'morgan_fts' in metrics:
        print(f"   Morgan FTS:    {metrics['morgan_fts']:.4f}")
    if 'maccs_fts' in metrics:
        print(f"   MACCS FTS:     {metrics['maccs_fts']:.4f}")
    if 'rdk_fts' in metrics:
        print(f"   RDKit FTS:     {metrics['rdk_fts']:.4f}")
    
    # Analysis
    print(f"\n💡 Analysis:")
    if metrics['validity'] < 0.5:
        print(f"   ⚠️  Low validity ({metrics['validity']:.1%}) - model needs more training")
    else:
        print(f"   ✅ Good validity ({metrics['validity']:.1%})")
    
    if metrics['uniqueness'] < 0.8:
        print(f"   ⚠️  Low diversity - model generates repetitive molecules")
    else:
        print(f"   ✅ Good molecular diversity")
    
    if metrics['exact_match'] < 0.1:
        print(f"   ⚠️  Poor scaffold preservation - model not following input scaffolds well")
    
    print("\n" + "="*60)
    print(f"Evaluation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return metrics

if __name__ == "__main__":
    results = main()