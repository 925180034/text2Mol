"""
Interactive Molecule Designer API

This module provides an interactive interface for molecular design
using the scaffold-based generation system.
"""

import logging
from typing import Dict, List, Optional, Any, Union
import json
from pathlib import Path

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw

from ..models.core_model import ScaffoldBasedMolT5Generator
from ..utils.scaffold_utils import ScaffoldExtractor
from ..utils.mol_utils import MolecularUtils
from ..utils.visualization import MolecularVisualizer

logger = logging.getLogger(__name__)


class InteractiveMoleculeDesigner:
    """
    Interactive interface for scaffold-based molecular design.
    
    This class provides a user-friendly API for generating molecules
    with specific scaffolds and properties.
    """
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        device: str = "auto",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the interactive molecule designer.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to use ('cpu', 'cuda', or 'auto')
            config: Model configuration dictionary
        """
        self.device = self._setup_device(device)
        self.config = config or {}
        
        # Initialize components
        self.model = None
        self.scaffold_extractor = ScaffoldExtractor()
        self.mol_processor = MolecularUtils()
        self.visualizer = MolecularVisualizer()
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
            
        logger.info(f"InteractiveMoleculeDesigner initialized on {self.device}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a trained model from checkpoint.
        
        Args:
            model_path: Path to model checkpoint
            
        Returns:
            bool: True if successful
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Initialize model with saved config
            model_config = checkpoint.get('config', self.config)
            self.model = ScaffoldBasedMolT5Generator(model_config)
            
            # Load state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def generate_molecules(
        self,
        text_prompt: str,
        scaffold_smiles: Optional[str] = None,
        num_molecules: int = 5,
        max_length: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        return_scores: bool = False
    ) -> Dict[str, Any]:
        """
        Generate molecules based on text prompt and optional scaffold.
        
        Args:
            text_prompt: Natural language description
            scaffold_smiles: Optional scaffold SMILES
            num_molecules: Number of molecules to generate
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            return_scores: Whether to return generation scores
            
        Returns:
            Dict containing generated molecules and metadata
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        try:
            # Prepare inputs
            inputs = {
                'text': text_prompt,
                'scaffold_smiles': scaffold_smiles
            }
            
            # Generation parameters
            gen_params = {
                'num_molecules': num_molecules,
                'max_length': max_length,
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p,
                'return_scores': return_scores
            }
            
            # Generate molecules
            with torch.no_grad():
                results = self.model.generate(inputs, **gen_params)
            
            # Process results
            generated_molecules = []
            for i, smiles in enumerate(results['smiles']):
                mol_data = {
                    'smiles': smiles,
                    'mol': Chem.MolFromSmiles(smiles) if smiles else None,
                    'valid': smiles is not None and Chem.MolFromSmiles(smiles) is not None
                }
                
                if return_scores and 'scores' in results:
                    mol_data['score'] = results['scores'][i]
                
                generated_molecules.append(mol_data)
            
            # Calculate statistics
            valid_count = sum(1 for mol in generated_molecules if mol['valid'])
            validity_rate = valid_count / len(generated_molecules) if generated_molecules else 0
            
            return {
                'molecules': generated_molecules,
                'input_prompt': text_prompt,
                'scaffold_smiles': scaffold_smiles,
                'generation_params': gen_params,
                'statistics': {
                    'total_generated': len(generated_molecules),
                    'valid_molecules': valid_count,
                    'validity_rate': validity_rate
                }
            }
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def analyze_scaffold(
        self, 
        smiles: str,
        scaffold_type: str = "murcko"
    ) -> Dict[str, Any]:
        """
        Analyze molecular scaffold from SMILES.
        
        Args:
            smiles: Input SMILES string
            scaffold_type: Type of scaffold to extract
            
        Returns:
            Dict containing scaffold analysis
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            
            # Extract scaffold
            scaffold_result = self.scaffold_extractor.extract_scaffold(
                mol, scaffold_type=scaffold_type
            )
            
            return {
                'input_smiles': smiles,
                'scaffold_smiles': scaffold_result['scaffold_smiles'],
                'scaffold_type': scaffold_type,
                'num_atoms': mol.GetNumAtoms(),
                'num_bonds': mol.GetNumBonds(),
                'scaffold_atoms': scaffold_result.get('num_scaffold_atoms', 0),
                'side_chains': scaffold_result.get('side_chains', [])
            }
            
        except Exception as e:
            logger.error(f"Scaffold analysis failed: {e}")
            raise
    
    def visualize_molecules(
        self,
        molecules: List[str],
        output_path: Optional[str] = None,
        mol_per_row: int = 4,
        img_size: tuple = (300, 300)
    ) -> str:
        """
        Visualize molecules in a grid.
        
        Args:
            molecules: List of SMILES strings
            output_path: Optional output path for image
            mol_per_row: Number of molecules per row
            img_size: Size of each molecule image
            
        Returns:
            str: Path to generated image
        """
        try:
            return self.visualizer.draw_molecules_grid(
                molecules=molecules,
                output_path=output_path,
                mol_per_row=mol_per_row,
                img_size=img_size
            )
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            raise
    
    def save_results(
        self,
        results: Dict[str, Any],
        output_path: str,
        format: str = "json"
    ) -> bool:
        """
        Save generation results to file.
        
        Args:
            results: Results dictionary from generate_molecules
            output_path: Output file path
            format: Output format ('json', 'csv', 'sdf')
            
        Returns:
            bool: True if successful
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == "json":
                # Prepare JSON-serializable data
                json_data = {
                    'input_prompt': results['input_prompt'],
                    'scaffold_smiles': results['scaffold_smiles'],
                    'generation_params': results['generation_params'],
                    'statistics': results['statistics'],
                    'molecules': [
                        {
                            'smiles': mol['smiles'],
                            'valid': mol['valid'],
                            'score': mol.get('score', None)
                        }
                        for mol in results['molecules']
                    ]
                }
                
                with open(output_path, 'w') as f:
                    json.dump(json_data, f, indent=2)
                    
            elif format.lower() == "csv":
                import pandas as pd
                
                df_data = []
                for mol in results['molecules']:
                    df_data.append({
                        'smiles': mol['smiles'],
                        'valid': mol['valid'],
                        'score': mol.get('score', None)
                    })
                
                df = pd.DataFrame(df_data)
                df.to_csv(output_path, index=False)
                
            elif format.lower() == "sdf":
                from rdkit.Chem import SDWriter
                
                writer = SDWriter(str(output_path))
                for mol_data in results['molecules']:
                    if mol_data['mol'] is not None:
                        writer.write(mol_data['mol'])
                writer.close()
                
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Results saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict containing model information
        """
        if self.model is None:
            return {"status": "No model loaded"}
        
        return {
            "status": "Model loaded",
            "device": str(self.device),
            "config": self.config,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
        }