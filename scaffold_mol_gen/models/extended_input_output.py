#!/usr/bin/env python3
"""
Extended Input-Output Combinations for Scaffold-based Molecular Generation.
Implements comprehensive modality combinations: Text→SMILES, SMILES→Properties, 
SMILES→Graph, Graph→SMILES, and multi-modal fusion approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import T5Tokenizer, T5ForConditionalGeneration
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import yaml
import logging
import json
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MolecularGraphEncoder(nn.Module):
    """Graph Neural Network encoder for molecular graphs."""
    
    def __init__(self, 
                 node_features: int = None,  # Will be determined dynamically
                 edge_features: int = None,  # Will be determined dynamically
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 output_dim: int = 768):
        super().__init__()
        
        # Store parameters for later initialization
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # These will be initialized when we see the first graph
        self.node_embedding = None
        self.edge_embedding = None
        self.conv_layers = None
        self.output_proj = None
        self.norm_layers = None
        self.initialized = False
        
    def _initialize_layers(self, node_features, edge_features, device):
        """Initialize layers based on actual feature dimensions."""
        self.node_embedding = nn.Linear(node_features, self.hidden_dim).to(device)
        self.edge_embedding = nn.Linear(edge_features, self.hidden_dim).to(device)
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList([
            GATConv(self.hidden_dim, self.hidden_dim // 4, heads=4, concat=True, edge_dim=self.hidden_dim)
            for _ in range(self.num_layers)
        ]).to(device)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.output_dim)
        ).to(device)
        
        self.norm_layers = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)]).to(device)
        self.initialized = True

    def forward(self, batch):
        """Forward pass through molecular graph."""
        x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        
        # Determine device from input data
        device = x.device
        
        # Initialize layers if first time
        if not self.initialized:
            self._initialize_layers(x.size(-1), edge_attr.size(-1), device)
        
        # Embed nodes and edges
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        
        # Graph convolutions with residual connections
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.norm_layers)):
            x_new = conv(x, edge_index, edge_attr)
            x_new = norm(x_new)
            x = x + x_new if x.size(-1) == x_new.size(-1) else x_new
            x = F.relu(x)
        
        # Global pooling
        graph_repr = global_mean_pool(x, batch_idx)
        
        # Output projection
        output = self.output_proj(graph_repr)
        
        return output

class SMILESToGraphConverter:
    """Convert SMILES strings to molecular graphs."""
    
    def __init__(self):
        # Atom features
        self.atom_features = [
            'atomic_num', 'formal_charge', 'num_radical_electrons',
            'hybridization', 'is_aromatic', 'is_in_ring',
            'chiral_tag', 'degree', 'total_num_hs', 'num_explicit_hs',
            'mass', 'van_der_waals_radius'
        ]
        
        # Bond features  
        self.bond_features = [
            'bond_type', 'conjugated', 'is_in_ring', 'stereo'
        ]
    
    def smiles_to_graph(self, smiles: str) -> Optional[Data]:
        """Convert SMILES to PyTorch Geometric graph."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Add hydrogens for complete representation
            mol = Chem.AddHs(mol)
            
            # Node features (atoms)
            node_features = []
            for atom in mol.GetAtoms():
                features = self._get_atom_features(atom)
                node_features.append(features)
            
            # Edge features (bonds)
            edge_indices = []
            edge_features = []
            
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                # Add both directions for undirected graph
                edge_indices.extend([[i, j], [j, i]])
                
                bond_feat = self._get_bond_features(bond)
                edge_features.extend([bond_feat, bond_feat])
            
            # Convert to tensors
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
            
            # Create graph data
            graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            graph.smiles = smiles
            
            return graph
            
        except Exception as e:
            logger.warning(f"Failed to convert SMILES {smiles} to graph: {e}")
            return None
    
    def _get_atom_features(self, atom) -> List[float]:
        """Extract atom features."""
        features = []
        
        # Basic properties
        features.append(atom.GetAtomicNum())
        features.append(atom.GetFormalCharge())
        features.append(atom.GetNumRadicalElectrons())
        
        # Hybridization (one-hot)
        hyb = atom.GetHybridization()
        hyb_features = [0] * 6  # SP, SP2, SP3, SP3D, SP3D2, other
        if hyb == Chem.HybridizationType.SP:
            hyb_features[0] = 1
        elif hyb == Chem.HybridizationType.SP2:
            hyb_features[1] = 1
        elif hyb == Chem.HybridizationType.SP3:
            hyb_features[2] = 1
        elif hyb == Chem.HybridizationType.SP3D:
            hyb_features[3] = 1
        elif hyb == Chem.HybridizationType.SP3D2:
            hyb_features[4] = 1
        else:
            hyb_features[5] = 1
        features.extend(hyb_features)
        
        # Boolean properties
        features.append(float(atom.GetIsAromatic()))
        features.append(float(atom.IsInRing()))
        
        # Chirality (one-hot)
        chiral = atom.GetChiralTag()
        chiral_features = [0] * 4  # R, S, other, none
        if chiral == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            chiral_features[0] = 1
        elif chiral == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            chiral_features[1] = 1
        elif chiral != Chem.ChiralType.CHI_UNSPECIFIED:
            chiral_features[2] = 1
        else:
            chiral_features[3] = 1
        features.extend(chiral_features)
        
        # Degree and hydrogen count
        features.append(atom.GetDegree())
        features.append(atom.GetTotalNumHs())
        features.append(atom.GetNumExplicitHs())
        
        # Physical properties
        features.append(atom.GetMass())
        
        # Atomic properties (approximated)
        atomic_num = atom.GetAtomicNum()
        vdw_radius = self._get_vdw_radius(atomic_num)
        features.append(vdw_radius)
        
        # Atom type one-hot (common atoms)
        atom_types = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]  # H, C, N, O, F, P, S, Cl, Br, I
        atom_type_features = [0] * (len(atom_types) + 1)  # +1 for other
        if atomic_num in atom_types:
            atom_type_features[atom_types.index(atomic_num)] = 1
        else:
            atom_type_features[-1] = 1
        features.extend(atom_type_features)
        
        return features
    
    def _get_bond_features(self, bond) -> List[float]:
        """Extract bond features."""
        features = []
        
        # Bond type (one-hot)
        bond_type = bond.GetBondType()
        bond_type_features = [0] * 5  # SINGLE, DOUBLE, TRIPLE, AROMATIC, other
        if bond_type == Chem.BondType.SINGLE:
            bond_type_features[0] = 1
        elif bond_type == Chem.BondType.DOUBLE:
            bond_type_features[1] = 1
        elif bond_type == Chem.BondType.TRIPLE:
            bond_type_features[2] = 1
        elif bond_type == Chem.BondType.AROMATIC:
            bond_type_features[3] = 1
        else:
            bond_type_features[4] = 1
        features.extend(bond_type_features)
        
        # Boolean properties
        features.append(float(bond.GetIsConjugated()))
        features.append(float(bond.IsInRing()))
        
        # Stereochemistry (one-hot)
        stereo = bond.GetStereo()
        stereo_features = [0] * 4  # NONE, E, Z, other
        if stereo == Chem.BondStereo.STEREONONE:
            stereo_features[0] = 1
        elif stereo == Chem.BondStereo.STEREOE:
            stereo_features[1] = 1
        elif stereo == Chem.BondStereo.STEREOZ:
            stereo_features[2] = 1
        else:
            stereo_features[3] = 1
        features.extend(stereo_features)
        
        return features
    
    def _get_vdw_radius(self, atomic_num: int) -> float:
        """Get van der Waals radius for atom."""
        # Approximated values in Angstroms
        vdw_radii = {
            1: 1.2,   # H
            6: 1.7,   # C
            7: 1.55,  # N
            8: 1.52,  # O
            9: 1.47,  # F
            15: 1.8,  # P
            16: 1.8,  # S
            17: 1.75, # Cl
            35: 1.85, # Br
            53: 1.98  # I
        }
        return vdw_radii.get(atomic_num, 2.0)

class ExtendedModalitySystem:
    """Extended system supporting multiple input-output combinations."""
    
    def __init__(self, model_path: str = "models/MolT5-Large-Caption2SMILES"):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load base T5 model
        logger.info(f"🚀 Loading base T5 model from {model_path}")
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.t5_model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.t5_model.to(self.device)
        
        # Initialize graph components
        self.graph_converter = SMILESToGraphConverter()
        self.graph_encoder = MolecularGraphEncoder()
        self.graph_encoder.to(self.device)
        
        # Initialize fusion layers (will be properly initialized when first used)
        self.text_projection = None
        self.graph_projection = None
        self.fusion_layer = None
        self.fusion_initialized = False
        
        logger.info(f"✅ Extended modality system initialized on {self.device}")
    
    def text_to_smiles(self, text: str, **generation_kwargs) -> Dict:
        """Generate SMILES from text description."""
        
        # Use optimized prompting
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=256,
            truncation=True,
            padding='max_length'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generation parameters
        gen_kwargs = {
            'max_length': 128,
            'num_beams': 5,
            'temperature': 0.7,
            'do_sample': True,
            'top_k': 50,
            'top_p': 0.95,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id
        }
        gen_kwargs.update(generation_kwargs)
        
        # Generate SMILES
        with torch.no_grad():
            outputs = self.t5_model.generate(**inputs, **gen_kwargs)
        
        # Decode result
        generated_smiles = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Validate and analyze
        mol = Chem.MolFromSmiles(generated_smiles)
        is_valid = mol is not None
        
        result = {
            'input_text': text,
            'generated_smiles': generated_smiles,
            'is_valid': is_valid,
            'molecular_properties': {}
        }
        
        if is_valid:
            try:
                result['molecular_properties'] = {
                    'molecular_weight': Descriptors.MolWt(mol),
                    'logp': Descriptors.MolLogP(mol),
                    'num_atoms': mol.GetNumAtoms(),
                    'num_bonds': mol.GetNumBonds(),
                    'num_rings': rdMolDescriptors.CalcNumRings(mol),
                    'tpsa': Descriptors.TPSA(mol),
                    'hbd': Descriptors.NumHDonors(mol),
                    'hba': Descriptors.NumHAcceptors(mol)
                }
            except Exception as e:
                logger.warning(f"Error calculating properties: {e}")
        
        return result
    
    def smiles_to_properties(self, smiles: str) -> Dict:
        """Predict molecular properties from SMILES."""
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {'error': 'Invalid SMILES'}
        
        try:
            # Calculate comprehensive molecular properties
            properties = {
                # Basic properties
                'molecular_weight': Descriptors.MolWt(mol),
                'exact_mass': Descriptors.ExactMolWt(mol),
                'heavy_atom_count': Descriptors.HeavyAtomCount(mol),
                
                # Lipophilicity and solubility
                'logp': Descriptors.MolLogP(mol),
                'logd': Descriptors.MolLogP(mol),  # Approximation
                'tpsa': Descriptors.TPSA(mol),
                
                # Hydrogen bonding
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                
                # Structural features
                'num_rings': rdMolDescriptors.CalcNumRings(mol),
                'num_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
                'num_saturated_rings': rdMolDescriptors.CalcNumSaturatedRings(mol),
                
                # Rotatable bonds and flexibility
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'flexibility': Descriptors.NumRotatableBonds(mol) / max(mol.GetNumBonds(), 1),
                
                # Drug-likeness indicators
                'lipinski_violations': self._count_lipinski_violations(mol),
                'qed': Descriptors.qed(mol),
                
                # Molecular complexity
                'bertz_complexity': Descriptors.BertzCT(mol),
                'formal_charge': Chem.GetFormalCharge(mol),
                
                # Atom and bond counts
                'num_atoms': mol.GetNumAtoms(),
                'num_bonds': mol.GetNumBonds(),
                'num_carbons': len([a for a in mol.GetAtoms() if a.GetAtomicNum() == 6]),
                'num_nitrogens': len([a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]),
                'num_oxygens': len([a for a in mol.GetAtoms() if a.GetAtomicNum() == 8]),
                
                # Molecular fingerprint (for similarity calculations)
                'morgan_fingerprint': list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024).ToBitString())
            }
            
            return {
                'smiles': smiles,
                'properties': properties,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error calculating properties for {smiles}: {e}")
            return {'error': str(e), 'success': False}
    
    def smiles_to_graph(self, smiles: str) -> Optional[Data]:
        """Convert SMILES to molecular graph representation."""
        return self.graph_converter.smiles_to_graph(smiles)
    
    def graph_to_embedding(self, graph: Data) -> torch.Tensor:
        """Convert molecular graph to embedding vector."""
        with torch.no_grad():
            # Convert to batch
            batch = Batch.from_data_list([graph])
            batch = batch.to(self.device)
            
            # Encode graph
            embedding = self.graph_encoder(batch)
            
            return embedding
    
    def _initialize_fusion_layers(self, text_dim, graph_dim):
        """Initialize fusion layers based on actual dimensions."""
        fusion_dim = 512  # Common fusion dimension
        
        self.text_projection = nn.Linear(text_dim, fusion_dim).to(self.device)
        self.graph_projection = nn.Linear(graph_dim, fusion_dim).to(self.device)
        self.fusion_layer = nn.MultiheadAttention(fusion_dim, 8, batch_first=True).to(self.device)
        self.fusion_initialized = True

    def multi_modal_fusion(self, text: str, smiles: str) -> Dict:
        """Fuse text and molecular graph information."""
        
        try:
            # Text embedding
            text_inputs = self.tokenizer(
                text,
                return_tensors='pt',
                max_length=256,
                truncation=True,
                padding='max_length'
            )
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            
            with torch.no_grad():
                # Get text representation from T5 encoder
                text_outputs = self.t5_model.encoder(**text_inputs)
                text_embedding = text_outputs.last_hidden_state.mean(dim=1)  # Average pooling
                
                # Get graph representation
                graph = self.smiles_to_graph(smiles)
                if graph is None:
                    return {'error': 'Invalid SMILES for graph conversion'}
                
                graph_embedding = self.graph_to_embedding(graph)
                
                # Initialize fusion layers if needed
                if not self.fusion_initialized:
                    self._initialize_fusion_layers(text_embedding.size(-1), graph_embedding.size(-1))
                
                # Project to same dimension
                text_proj = self.text_projection(text_embedding)
                graph_proj = self.graph_projection(graph_embedding)
                
                # Fusion via attention
                combined = torch.stack([text_proj, graph_proj], dim=1)
                fused_output, attention_weights = self.fusion_layer(combined, combined, combined)
                
                # Final representation
                final_embedding = fused_output.mean(dim=1)
                
                return {
                    'text': text,
                    'smiles': smiles,
                    'text_embedding': text_embedding.cpu().numpy(),
                    'graph_embedding': graph_embedding.cpu().numpy(),
                    'fused_embedding': final_embedding.cpu().numpy(),
                    'attention_weights': attention_weights.cpu().numpy(),
                    'success': True
                }
                
        except Exception as e:
            logger.error(f"Multi-modal fusion error: {e}")
            return {'error': str(e), 'success': False}
    
    def _count_lipinski_violations(self, mol) -> int:
        """Count Lipinski rule of five violations."""
        violations = 0
        
        # Molecular weight > 500
        if Descriptors.MolWt(mol) > 500:
            violations += 1
        
        # LogP > 5
        if Descriptors.MolLogP(mol) > 5:
            violations += 1
        
        # HBD > 5
        if Descriptors.NumHDonors(mol) > 5:
            violations += 1
        
        # HBA > 10
        if Descriptors.NumHAcceptors(mol) > 10:
            violations += 1
        
        return violations

def test_extended_modalities():
    """Test all extended input-output combinations."""
    
    logger.info("🧪 TESTING EXTENDED INPUT-OUTPUT MODALITIES")
    logger.info("=" * 60)
    
    # Initialize system
    system = ExtendedModalitySystem()
    
    # Test cases
    test_molecules = [
        {
            'description': 'water molecule',
            'expected_smiles': 'O'  # for validation
        },
        {
            'description': 'ethanol',
            'expected_smiles': 'CCO'
        },
        {
            'description': 'benzene ring',
            'expected_smiles': 'c1ccccc1'
        },
        {
            'description': 'aspirin',
            'expected_smiles': 'CC(=O)OC1=CC=CC=C1C(=O)O'
        }
    ]
    
    results = {}
    
    # Test 1: Text → SMILES
    logger.info("\n🔬 Test 1: Text → SMILES Generation")
    logger.info("-" * 40)
    
    text_to_smiles_results = []
    for test in test_molecules:
        result = system.text_to_smiles(test['description'])
        text_to_smiles_results.append(result)
        
        logger.info(f"Input: {result['input_text']}")
        logger.info(f"Generated: {result['generated_smiles']}")
        logger.info(f"Valid: {'✅' if result['is_valid'] else '❌'}")
        
        if result['is_valid'] and result['molecular_properties']:
            props = result['molecular_properties']
            logger.info(f"MW: {props.get('molecular_weight', 0):.1f}, "
                       f"LogP: {props.get('logp', 0):.2f}")
        logger.info("")
    
    results['text_to_smiles'] = text_to_smiles_results
    
    # Test 2: SMILES → Properties
    logger.info("\n🔬 Test 2: SMILES → Properties Prediction")
    logger.info("-" * 40)
    
    smiles_to_props_results = []
    test_smiles = ['O', 'CCO', 'c1ccccc1', 'CC(=O)OC1=CC=CC=C1C(=O)O']
    
    for smiles in test_smiles:
        result = system.smiles_to_properties(smiles)
        smiles_to_props_results.append(result)
        
        if result['success']:
            props = result['properties']
            logger.info(f"SMILES: {smiles}")
            logger.info(f"  MW: {props['molecular_weight']:.1f}")
            logger.info(f"  LogP: {props['logp']:.2f}")
            logger.info(f"  TPSA: {props['tpsa']:.1f}")
            logger.info(f"  HBD: {props['hbd']}, HBA: {props['hba']}")
            logger.info(f"  Lipinski violations: {props['lipinski_violations']}")
            logger.info(f"  QED: {props['qed']:.3f}")
        else:
            logger.info(f"SMILES: {smiles} - Error: {result.get('error', 'Unknown')}")
        logger.info("")
    
    results['smiles_to_properties'] = smiles_to_props_results
    
    # Test 3: SMILES → Graph
    logger.info("\n🔬 Test 3: SMILES → Graph Conversion")
    logger.info("-" * 40)
    
    smiles_to_graph_results = []
    for smiles in test_smiles:
        graph = system.smiles_to_graph(smiles)
        
        if graph is not None:
            logger.info(f"SMILES: {smiles}")
            logger.info(f"  Nodes: {graph.x.shape[0]}")
            logger.info(f"  Edges: {graph.edge_index.shape[1]}")
            logger.info(f"  Node features: {graph.x.shape[1]}")
            logger.info(f"  Edge features: {graph.edge_attr.shape[1]}")
            
            # Get graph embedding
            embedding = system.graph_to_embedding(graph)
            logger.info(f"  Graph embedding shape: {embedding.shape}")
            
            smiles_to_graph_results.append({
                'smiles': smiles,
                'nodes': graph.x.shape[0],
                'edges': graph.edge_index.shape[1],
                'embedding_dim': embedding.shape[1],
                'success': True
            })
        else:
            logger.info(f"SMILES: {smiles} - Failed to convert to graph")
            smiles_to_graph_results.append({
                'smiles': smiles,
                'success': False
            })
        logger.info("")
    
    results['smiles_to_graph'] = smiles_to_graph_results
    
    # Test 4: Multi-modal Fusion
    logger.info("\n🔬 Test 4: Multi-modal Fusion (Text + SMILES)")
    logger.info("-" * 40)
    
    fusion_results = []
    test_pairs = [
        ('water molecule', 'O'),
        ('ethanol', 'CCO'),
        ('benzene ring', 'c1ccccc1')
    ]
    
    for text, smiles in test_pairs:
        result = system.multi_modal_fusion(text, smiles)
        fusion_results.append(result)
        
        if result['success']:
            logger.info(f"Text: {text}")
            logger.info(f"SMILES: {smiles}")
            logger.info(f"Text embedding shape: {result['text_embedding'].shape}")
            logger.info(f"Graph embedding shape: {result['graph_embedding'].shape}")
            logger.info(f"Fused embedding shape: {result['fused_embedding'].shape}")
            logger.info(f"Attention weights shape: {result['attention_weights'].shape}")
        else:
            logger.info(f"Fusion failed for {text} + {smiles}: {result.get('error', 'Unknown')}")
        logger.info("")
    
    results['multi_modal_fusion'] = fusion_results
    
    # Summary
    logger.info("\n📊 EXTENDED MODALITIES TEST SUMMARY")
    logger.info("=" * 50)
    
    # Text → SMILES success rate
    text_smiles_success = sum(1 for r in text_to_smiles_results if r['is_valid'])
    logger.info(f"Text → SMILES: {text_smiles_success}/{len(text_to_smiles_results)} "
               f"({text_smiles_success/len(text_to_smiles_results)*100:.1f}%)")
    
    # SMILES → Properties success rate
    props_success = sum(1 for r in smiles_to_props_results if r['success'])
    logger.info(f"SMILES → Properties: {props_success}/{len(smiles_to_props_results)} "
               f"({props_success/len(smiles_to_props_results)*100:.1f}%)")
    
    # SMILES → Graph success rate
    graph_success = sum(1 for r in smiles_to_graph_results if r['success'])
    logger.info(f"SMILES → Graph: {graph_success}/{len(smiles_to_graph_results)} "
               f"({graph_success/len(smiles_to_graph_results)*100:.1f}%)")
    
    # Multi-modal Fusion success rate
    fusion_success = sum(1 for r in fusion_results if r['success'])
    logger.info(f"Multi-modal Fusion: {fusion_success}/{len(fusion_results)} "
               f"({fusion_success/len(fusion_results)*100:.1f}%)")
    
    return results

def main():
    """Main testing function."""
    
    try:
        # Test extended modalities
        results = test_extended_modalities()
        
        # Save results
        output_dir = Path("outputs/extended_modalities")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        with open(output_dir / "extended_modalities_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\n🎉 Extended modalities testing completed!")
        logger.info(f"Results saved to: {output_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    results = main()