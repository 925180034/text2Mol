#!/usr/bin/env python3
"""
🧪 9-Modality Molecular Generation System - Comprehensive Evaluation

This script tests all 9 input-output combinations:
(SMILES/Graph/Image scaffold) × (SMILES/Graph/Image output) = 9 combinations

Author: Text2Mol Development Team
Date: 2025-08-16
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging
import time
import json
from datetime import datetime
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import model components
from scaffold_mol_gen.models.end2end_model import End2EndMolecularGenerator
from scaffold_mol_gen.training.metrics import GenerationMetrics
from scaffold_mol_gen.utils.mol_utils import MolecularUtils

# Import RDKit for molecular operations
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, Crippen
from rdkit.Chem import AllChem
from PIL import Image
import torch_geometric
from torch_geometric.data import Data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('nine_modality_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ModalityResult:
    """Results for a single modality combination"""
    input_modality: str
    output_modality: str
    num_samples: int
    success_rate: float
    validity: float
    uniqueness: float
    novelty: float
    similarity_maccs: float
    similarity_morgan: float
    similarity_rdk: float
    avg_generation_time: float
    errors: List[str]


class NineModalityEvaluator:
    """
    Comprehensive evaluator for 9-modality molecular generation system
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cuda'):
        """
        Initialize the evaluator
        
        Args:
            model_path: Path to trained model checkpoint
            device: Computing device ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"🖥️ Using device: {self.device}")
        
        # Model initialization
        self.model = None
        self.model_path = model_path
        
        # Metrics calculator
        self.metrics = GenerationMetrics()
        
        # Define all 9 modality combinations
        self.modality_combinations = [
            ('smiles', 'smiles'), ('smiles', 'graph'), ('smiles', 'image'),
            ('graph', 'smiles'),  ('graph', 'graph'),  ('graph', 'image'),
            ('image', 'smiles'),  ('image', 'graph'),  ('image', 'image')
        ]
        
        # Results storage
        self.results = {}
        
        # Test data
        self.test_data = None
        
    def load_model(self) -> bool:
        """Load the trained model"""
        try:
            if self.model_path and Path(self.model_path).exists():
                logger.info(f"📦 Loading model from: {self.model_path}")
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # Initialize model
                self.model = End2EndMolecularGenerator(
                    hidden_size=768,
                    molt5_path="laituan245/molt5-large-caption2smiles",  # Use HuggingFace model ID
                    use_scibert=False,
                    freeze_encoders=True,
                    freeze_molt5=False,
                    device=str(self.device)
                )
                
                # Load state dict
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                
                self.model.to(self.device)
                self.model.eval()
                logger.info("✅ Model loaded successfully")
                return True
            else:
                logger.warning("⚠️ No model checkpoint found, using untrained model")
                self.model = End2EndMolecularGenerator(
                    hidden_size=768,
                    molt5_path="laituan245/molt5-large-caption2smiles",  # Use HuggingFace model ID
                    use_scibert=False,
                    freeze_encoders=True,
                    freeze_molt5=False,
                    device=str(self.device)
                )
                self.model.to(self.device)
                self.model.eval()
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            return False
    
    def load_test_data(self, data_path: str = "Datasets/test.csv", num_samples: int = 100):
        """
        Load test data from ChEBI-20 dataset
        
        Args:
            data_path: Path to test data CSV
            num_samples: Number of samples to test
        """
        try:
            logger.info(f"📂 Loading test data from: {data_path}")
            df = pd.read_csv(data_path)
            
            # Sample data if needed
            if len(df) > num_samples:
                df = df.sample(n=num_samples, random_state=42)
            
            self.test_data = {
                'scaffolds': df['scaffold'].tolist(),
                'texts': df['text'].tolist(),
                'targets': df['SMILES'].tolist()
            }
            
            logger.info(f"✅ Loaded {len(df)} test samples")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load test data: {str(e)}")
            # Use synthetic test data
            self.test_data = self._create_synthetic_test_data()
            return True
    
    def _create_synthetic_test_data(self) -> Dict:
        """Create synthetic test data for demonstration"""
        return {
            'scaffolds': [
                'c1ccccc1',  # Benzene
                'c1ccc2c(c1)cccc2',  # Naphthalene
                'c1ccc2c(c1)[nH]c3ccccc32',  # Indole
                'C1CCC2CCCCC2C1',  # Decalin
                'c1cccnc1'  # Pyridine
            ],
            'texts': [
                'Anti-inflammatory drug with carboxylic acid group',
                'Antibiotic compound with amino and hydroxyl groups',
                'Antiviral agent with multiple hydroxyl groups',
                'Pain relief medication with ester linkage',
                'Cardiovascular drug with nitrogen heterocycle'
            ],
            'targets': [
                'CC(C)Cc1ccc(cc1)C(C)C(=O)O',  # Ibuprofen-like
                'CC1CC(C)C(O)C(C)C1N',  # Antibiotic-like
                'CC(C)C1CCC(O)CC1O',  # Antiviral-like
                'CC(=O)Oc1ccccc1C(=O)O',  # Aspirin-like
                'c1ccc2c(c1)c(cn2)CCN'  # Cardiovascular-like
            ]
        }
    
    def _convert_scaffold_modality(self, scaffold: str, target_modality: str) -> Any:
        """
        Convert scaffold from SMILES to target modality
        
        Args:
            scaffold: SMILES string
            target_modality: Target modality ('smiles', 'graph', 'image')
        """
        try:
            if target_modality == 'smiles':
                return scaffold
            
            mol = Chem.MolFromSmiles(scaffold)
            if mol is None:
                logger.warning(f"Invalid SMILES: {scaffold}")
                return None
            
            if target_modality == 'graph':
                # Convert to PyTorch Geometric graph
                return self._mol_to_graph(mol)
            
            elif target_modality == 'image':
                # Convert to image
                return self._mol_to_image(mol)
            
        except Exception as e:
            logger.error(f"Failed to convert scaffold: {str(e)}")
            return None
    
    def _mol_to_graph(self, mol) -> Data:
        """Convert RDKit mol to PyTorch Geometric graph"""
        # Get atom features
        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                atom.GetMass(),
                int(atom.IsInRing()),
                int(atom.GetChiralTag()),
                atom.GetTotalNumHs()
            ]
            atom_features.append(features)
        
        # Get edge indices and features
        edge_indices = []
        edge_features = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.extend([[i, j], [j, i]])  # Bidirectional
            
            bond_features = [
                int(bond.GetBondType()),
                int(bond.GetIsConjugated()),
                int(bond.IsInRing()),
                int(bond.GetStereo())
            ]
            edge_features.extend([bond_features, bond_features])
        
        # Create PyTorch Geometric Data object
        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float) if edge_features else None
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def _mol_to_image(self, mol, size=(299, 299)) -> np.ndarray:
        """Convert RDKit mol to image array"""
        img = Draw.MolToImage(mol, size=size)
        return np.array(img) / 255.0  # Normalize to [0, 1]
    
    def evaluate_modality_combination(self, 
                                     input_modality: str, 
                                     output_modality: str) -> ModalityResult:
        """
        Evaluate a single modality combination
        
        Args:
            input_modality: Input scaffold modality
            output_modality: Output modality
        """
        logger.info(f"\n🔬 Evaluating: {input_modality} → {output_modality}")
        logger.info("=" * 50)
        
        results = {
            'input_modality': input_modality,
            'output_modality': output_modality,
            'num_samples': len(self.test_data['scaffolds']),
            'success_rate': 0.0,
            'validity': 0.0,
            'uniqueness': 0.0,
            'novelty': 0.0,
            'similarity_maccs': 0.0,
            'similarity_morgan': 0.0,
            'similarity_rdk': 0.0,
            'avg_generation_time': 0.0,
            'errors': []
        }
        
        generated_molecules = []
        generation_times = []
        successful_generations = 0
        
        # Process each test sample
        for idx, (scaffold, text, target) in enumerate(zip(
            self.test_data['scaffolds'],
            self.test_data['texts'],
            self.test_data['targets']
        )):
            try:
                # Convert scaffold to input modality
                scaffold_input = self._convert_scaffold_modality(scaffold, input_modality)
                if scaffold_input is None:
                    results['errors'].append(f"Sample {idx}: Failed to convert scaffold")
                    continue
                
                # Generate molecule
                start_time = time.time()
                
                with torch.no_grad():
                    output = self.model(
                        scaffold_data=scaffold_input,
                        text_data=text,
                        scaffold_modality=input_modality,
                        output_modality=output_modality
                    )
                
                generation_time = time.time() - start_time
                generation_times.append(generation_time)
                
                # Extract generated molecule based on output modality
                if output_modality == 'smiles':
                    if 'generated_smiles' in output:
                        generated = output['generated_smiles']
                        if isinstance(generated, list):
                            generated = generated[0]
                        generated_molecules.append(generated)
                        successful_generations += 1
                    else:
                        results['errors'].append(f"Sample {idx}: No SMILES generated")
                        
                elif output_modality == 'graph':
                    if f'generated_{output_modality}' in output:
                        # For graph output, we need to verify it's a valid graph
                        generated_graph = output[f'generated_{output_modality}']
                        if generated_graph is not None:
                            generated_molecules.append(generated_graph)
                            successful_generations += 1
                    else:
                        results['errors'].append(f"Sample {idx}: No graph generated")
                        
                elif output_modality == 'image':
                    if f'generated_{output_modality}' in output:
                        # For image output, verify it's a valid image array
                        generated_image = output[f'generated_{output_modality}']
                        if generated_image is not None:
                            generated_molecules.append(generated_image)
                            successful_generations += 1
                    else:
                        results['errors'].append(f"Sample {idx}: No image generated")
                
            except Exception as e:
                results['errors'].append(f"Sample {idx}: {str(e)}")
                continue
        
        # Calculate metrics
        results['success_rate'] = (successful_generations / len(self.test_data['scaffolds'])) * 100
        
        if generation_times:
            results['avg_generation_time'] = np.mean(generation_times)
        
        # Calculate molecular metrics for SMILES output
        if output_modality == 'smiles' and generated_molecules:
            try:
                # Validity
                valid_mols = [Chem.MolFromSmiles(smi) for smi in generated_molecules if smi]
                results['validity'] = (len([m for m in valid_mols if m is not None]) / len(generated_molecules)) * 100
                
                # Uniqueness
                unique_smiles = set(generated_molecules)
                results['uniqueness'] = (len(unique_smiles) / len(generated_molecules)) * 100
                
                # Novelty (compared to targets)
                novel_smiles = unique_smiles - set(self.test_data['targets'])
                results['novelty'] = (len(novel_smiles) / len(unique_smiles)) * 100 if unique_smiles else 0
                
                # Similarities (if we have valid molecules)
                valid_generated = [smi for smi in generated_molecules if Chem.MolFromSmiles(smi) is not None]
                valid_targets = [smi for smi in self.test_data['targets'] if Chem.MolFromSmiles(smi) is not None]
                
                if valid_generated and valid_targets:
                    similarities = self.metrics.calculate_all_similarities(
                        valid_generated[:min(10, len(valid_generated))],
                        valid_targets[:min(10, len(valid_targets))]
                    )
                    results['similarity_maccs'] = similarities.get('maccs', 0.0)
                    results['similarity_morgan'] = similarities.get('morgan', 0.0)
                    results['similarity_rdk'] = similarities.get('rdk', 0.0)
                    
            except Exception as e:
                logger.warning(f"Failed to calculate metrics: {str(e)}")
        
        return ModalityResult(**results)
    
    def run_comprehensive_evaluation(self):
        """Run evaluation for all 9 modality combinations"""
        logger.info("\n" + "="*60)
        logger.info("🚀 Starting 9-Modality Comprehensive Evaluation")
        logger.info("="*60)
        
        # Load model
        if not self.load_model():
            logger.error("Failed to load model, aborting evaluation")
            return
        
        # Load test data
        if not self.load_test_data(num_samples=50):
            logger.error("Failed to load test data, aborting evaluation")
            return
        
        # Evaluate each modality combination
        for input_mod, output_mod in self.modality_combinations:
            try:
                result = self.evaluate_modality_combination(input_mod, output_mod)
                self.results[f"{input_mod}_to_{output_mod}"] = result
                
                # Log summary for this combination
                logger.info(f"\n✅ {input_mod} → {output_mod} Results:")
                logger.info(f"   Success Rate: {result.success_rate:.1f}%")
                logger.info(f"   Validity: {result.validity:.1f}%")
                logger.info(f"   Uniqueness: {result.uniqueness:.1f}%")
                logger.info(f"   Novelty: {result.novelty:.1f}%")
                logger.info(f"   Avg Time: {result.avg_generation_time:.3f}s")
                
            except Exception as e:
                logger.error(f"❌ Failed to evaluate {input_mod} → {output_mod}: {str(e)}")
                continue
        
        # Generate comprehensive report
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive evaluation report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = Path(f"evaluation_results/nine_modality_{timestamp}")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results as JSON
        results_dict = {k: asdict(v) for k, v in self.results.items()}
        with open(report_dir / "detailed_results.json", 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # Generate summary report
        summary_lines = [
            "# 9-Modality Evaluation Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary Statistics",
            ""
        ]
        
        # Calculate overall statistics
        all_success_rates = [r.success_rate for r in self.results.values()]
        all_validities = [r.validity for r in self.results.values() if r.validity > 0]
        all_times = [r.avg_generation_time for r in self.results.values() if r.avg_generation_time > 0]
        
        summary_lines.extend([
            f"- **Average Success Rate**: {np.mean(all_success_rates):.1f}%",
            f"- **Average Validity**: {np.mean(all_validities):.1f}%" if all_validities else "- **Average Validity**: N/A",
            f"- **Average Generation Time**: {np.mean(all_times):.3f}s" if all_times else "- **Average Generation Time**: N/A",
            "",
            "## Detailed Results by Modality",
            ""
        ])
        
        # Add detailed results table
        summary_lines.append("| Input | Output | Success | Validity | Uniqueness | Novelty | Time (s) |")
        summary_lines.append("|-------|--------|---------|----------|------------|---------|----------|")
        
        for key, result in self.results.items():
            summary_lines.append(
                f"| {result.input_modality} | {result.output_modality} | "
                f"{result.success_rate:.1f}% | {result.validity:.1f}% | "
                f"{result.uniqueness:.1f}% | {result.novelty:.1f}% | "
                f"{result.avg_generation_time:.3f} |"
            )
        
        # Performance matrix visualization
        summary_lines.extend([
            "",
            "## Performance Matrix",
            "",
            "```",
            "       Output →",
            "Input  SMILES  Graph  Image",
            "  ↓"
        ])
        
        for input_mod in ['smiles', 'graph', 'image']:
            row = f"{input_mod:7}"
            for output_mod in ['smiles', 'graph', 'image']:
                key = f"{input_mod}_to_{output_mod}"
                if key in self.results:
                    success = self.results[key].success_rate
                    row += f" {success:5.1f}%"
                else:
                    row += "   N/A "
            summary_lines.append(row)
        
        summary_lines.append("```")
        
        # Best and worst performing combinations
        if self.results:
            sorted_results = sorted(self.results.items(), key=lambda x: x[1].success_rate, reverse=True)
            
            summary_lines.extend([
                "",
                "## Best Performing Combinations",
                ""
            ])
            
            for key, result in sorted_results[:3]:
                summary_lines.append(f"- **{result.input_modality} → {result.output_modality}**: {result.success_rate:.1f}% success rate")
            
            summary_lines.extend([
                "",
                "## Areas for Improvement",
                ""
            ])
            
            for key, result in sorted_results[-3:]:
                if result.success_rate < 50:
                    summary_lines.append(f"- **{result.input_modality} → {result.output_modality}**: {result.success_rate:.1f}% success rate")
        
        # Save summary report
        with open(report_dir / "summary_report.md", 'w') as f:
            f.write('\n'.join(summary_lines))
        
        # Print summary to console
        logger.info("\n" + "="*60)
        logger.info("📊 EVALUATION COMPLETE")
        logger.info("="*60)
        logger.info(f"Results saved to: {report_dir}")
        logger.info("\nQuick Summary:")
        for line in summary_lines[4:20]:  # Print first part of summary
            if line and not line.startswith('#'):
                logger.info(line)


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='9-Modality Molecular Generation Evaluation')
    parser.add_argument('--model-path', type=str, 
                       default='/root/autodl-tmp/text2Mol-outputs/best_model.pt',
                       help='Path to trained model checkpoint')
    parser.add_argument('--num-samples', type=int, default=50,
                       help='Number of test samples to evaluate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Computing device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = NineModalityEvaluator(
        model_path=args.model_path,
        device=args.device
    )
    
    # Run comprehensive evaluation
    evaluator.run_comprehensive_evaluation()
    
    logger.info("\n✨ Evaluation complete!")


if __name__ == "__main__":
    main()