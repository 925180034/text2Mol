#!/usr/bin/env python3
"""
Test validity-focused model to ensure it generates valid SMILES
Quick validation before full training
"""

import os
import sys
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from rdkit import Chem
import pandas as pd
from tqdm import tqdm

# Add project path
sys.path.append('/root/text2Mol/scaffold-mol-generation')

def test_generation(model_path='/root/autodl-tmp/text2Mol-models/molt5-base'):
    """Test MolT5 model generation capability"""
    
    print("🧪 Testing MolT5-Base Model")
    print("="*50)
    
    # Load model and tokenizer
    print(f"Loading model from {model_path}")
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
        print("✅ Using GPU")
    else:
        print("⚠️ Using CPU")
    
    # Test cases with scaffold + text
    test_cases = [
        {
            'scaffold': 'c1ccccc1',  # Benzene ring
            'text': 'add hydroxyl group for solubility'
        },
        {
            'scaffold': 'CC(C)C',  # Isobutane
            'text': 'add carboxylic acid for polarity'
        },
        {
            'scaffold': 'c1ccc(cc1)N',  # Aniline
            'text': 'add methyl groups for lipophilicity'
        },
        {
            'scaffold': 'O=C(O)C',  # Acetic acid
            'text': 'extend carbon chain for increased hydrophobicity'
        },
        {
            'scaffold': 'c1cnccc1',  # Pyridine
            'text': 'add functional groups for drug-like properties'
        }
    ]
    
    print("\n📋 Test Cases:")
    print("-"*50)
    
    valid_count = 0
    results = []
    
    for i, test in enumerate(test_cases, 1):
        # Prepare input
        input_text = f"Generate molecule from scaffold: {test['scaffold']} with properties: {test['text']}"
        
        # Tokenize
        inputs = tokenizer(
            input_text,
            return_tensors='pt',
            max_length=256,
            truncation=True,
            padding=True
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate with different strategies
        print(f"\n🔬 Test Case {i}:")
        print(f"   Scaffold: {test['scaffold']}")
        print(f"   Text: {test['text']}")
        print(f"   Generated SMILES:")
        
        # Try different generation strategies
        strategies = [
            {'num_beams': 5, 'temperature': 0.8, 'top_p': 0.95, 'top_k': 50},
            {'num_beams': 3, 'temperature': 0.9, 'do_sample': True},
            {'num_beams': 1, 'temperature': 0.7, 'do_sample': True, 'top_k': 40}
        ]
        
        best_smiles = None
        best_validity = False
        
        for j, strategy in enumerate(strategies):
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=256,
                        repetition_penalty=1.2,
                        length_penalty=1.0,
                        early_stopping=True,
                        **strategy
                    )
                
                # Decode
                generated_smiles = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Validate
                try:
                    mol = Chem.MolFromSmiles(generated_smiles)
                    if mol is not None:
                        canonical_smiles = Chem.MolToSmiles(mol)
                        is_valid = True
                        if not best_validity:
                            best_smiles = canonical_smiles
                            best_validity = True
                    else:
                        is_valid = False
                        if best_smiles is None:
                            best_smiles = generated_smiles
                except:
                    is_valid = False
                    if best_smiles is None:
                        best_smiles = generated_smiles
                
                status = "✅ Valid" if is_valid else "❌ Invalid"
                print(f"      Strategy {j+1}: {generated_smiles[:50]}... [{status}]")
                
            except Exception as e:
                print(f"      Strategy {j+1}: Generation failed - {e}")
        
        # Store best result
        if best_validity:
            valid_count += 1
        
        results.append({
            'scaffold': test['scaffold'],
            'text': test['text'],
            'generated': best_smiles,
            'is_valid': best_validity
        })
    
    # Summary
    print("\n" + "="*50)
    print("📊 GENERATION SUMMARY")
    print("="*50)
    print(f"Total test cases: {len(test_cases)}")
    print(f"Valid generations: {valid_count}")
    print(f"Validity rate: {100*valid_count/len(test_cases):.1f}%")
    
    # Test on real data sample
    print("\n🔬 Testing on Real ChEBI-20 Data Sample...")
    print("-"*50)
    
    # Load a small sample from training data
    train_df = pd.read_csv('/root/text2Mol/scaffold-mol-generation/Datasets/train.csv')
    sample_df = train_df.head(10)
    
    real_valid_count = 0
    
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Testing"):
        # Prepare scaffold (simplified)
        scaffold = row['SMILES'][:20] if 'scaffold' not in row else row.get('scaffold', row['SMILES'][:20])
        text = row.get('text', row.get('description', 'molecule with drug-like properties'))
        
        input_text = f"Generate molecule from scaffold: {scaffold} with properties: {text}"
        
        inputs = tokenizer(
            input_text,
            return_tensors='pt',
            max_length=256,
            truncation=True,
            padding=True
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=256,
                    num_beams=5,
                    temperature=0.8,
                    top_p=0.95,
                    top_k=50,
                    repetition_penalty=1.2,
                    do_sample=True
                )
            
            generated_smiles = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Check validity
            try:
                mol = Chem.MolFromSmiles(generated_smiles)
                if mol is not None:
                    real_valid_count += 1
            except:
                pass
        except:
            pass
    
    print(f"\nReal data validity: {real_valid_count}/{len(sample_df)} ({100*real_valid_count/len(sample_df):.1f}%)")
    
    # Final assessment
    print("\n" + "="*50)
    print("🎯 MODEL ASSESSMENT")
    print("="*50)
    
    if valid_count >= len(test_cases) * 0.6:
        print("✅ Model shows good SMILES generation capability")
        print("   Ready for validity-focused training")
    else:
        print("⚠️ Model needs significant training to improve validity")
        print("   Validity-focused training is essential")
    
    return results

if __name__ == "__main__":
    results = test_generation()
    
    print("\n💡 Recommendations:")
    print("1. Use constrained beam search to enforce SMILES grammar")
    print("2. Implement validity loss to penalize invalid generations")
    print("3. Apply repetition penalty to prevent mode collapse")
    print("4. Use SMILES augmentation for robustness")
    print("5. Train in stages: validity → quality → refinement")