"""
Molecular graph conversion utilities.

This module provides tools for converting between SMILES strings and
graph representations for molecular data processing.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

logger = logging.getLogger(__name__)

class MolecularGraphConverter:
    """Convert between SMILES and graph representations."""
    
    def __init__(self):
        """Initialize graph converter."""
        self.atom_vocab = self._get_atom_vocabulary()
        self.bond_vocab = self._get_bond_vocabulary()
        
        # Cache for computed features
        self._atom_feature_cache = {}
        self._bond_feature_cache = {}
    
    def _get_atom_vocabulary(self) -> Dict[str, int]:
        """Get comprehensive atom type vocabulary."""
        atoms = [
            'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 
            'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 
            'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 
            'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', '[UNK]'
        ]
        return {atom: idx for idx, atom in enumerate(atoms)}
    
    def _get_bond_vocabulary(self) -> Dict[str, int]:
        """Get bond type vocabulary."""
        bonds = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
        return {bond: idx for idx, bond in enumerate(bonds)}
    
    def smiles_to_graph(self, smiles: str) -> Dict[str, Any]:
        """
        Convert SMILES to comprehensive graph representation.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Dictionary with graph data including nodes, edges, and metadata
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            
            # Add hydrogens for complete graph
            mol = Chem.AddHs(mol)
            
            # Node features (atoms)
            nodes = []
            atom_mapping = {}
            
            for idx, atom in enumerate(mol.GetAtoms()):
                atom_features = self._get_atom_features(atom)
                nodes.append(atom_features)
                atom_mapping[atom.GetIdx()] = idx
            
            # Edge features (bonds)
            edges = []
            edge_attrs = []
            
            for bond in mol.GetBonds():
                begin_idx = atom_mapping[bond.GetBeginAtomIdx()]
                end_idx = atom_mapping[bond.GetEndAtomIdx()]
                
                # Add both directions for undirected graph
                edges.extend([[begin_idx, end_idx], [end_idx, begin_idx]])
                
                bond_features = self._get_bond_features(bond)
                edge_attrs.extend([bond_features, bond_features])
            
            # Convert to numpy arrays
            nodes = np.array(nodes, dtype=np.float32)
            edges = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
            edge_attrs = np.array(edge_attrs, dtype=np.float32) if edge_attrs else np.zeros((0, 8), dtype=np.float32)
            
            # Additional graph properties
            graph_properties = self._compute_graph_properties(mol)
            
            return {
                'nodes': nodes,
                'edges': edges,
                'edge_attrs': edge_attrs,
                'num_nodes': len(nodes),
                'num_edges': len(edge_attrs),
                'smiles': smiles,
                'molecular_weight': graph_properties['molecular_weight'],
                'num_rings': graph_properties['num_rings'],
                'num_aromatic_rings': graph_properties['num_aromatic_rings'],
                'logp': graph_properties['logp'],
                'is_valid': True
            }
            
        except Exception as e:
            logger.error(f"Error converting SMILES to graph: {e}")
            # Return minimal valid graph for error handling
            return self._get_empty_graph(smiles)
    
    def _get_atom_features(self, atom) -> List[float]:
        """Extract comprehensive atom features."""
        features = []
        
        # Atom type (one-hot encoded)
        atom_type = atom.GetSymbol()
        atom_idx = self.atom_vocab.get(atom_type, self.atom_vocab['[UNK]'])
        atom_onehot = [0] * len(self.atom_vocab)
        atom_onehot[atom_idx] = 1
        features.extend(atom_onehot)
        
        # Chemical properties
        features.extend([
            atom.GetDegree(),                    # Number of bonds
            atom.GetFormalCharge(),              # Formal charge
            atom.GetHybridization().real,        # Hybridization
            atom.GetImplicitValence(),           # Implicit valence
            int(atom.GetIsAromatic()),           # Aromaticity
            atom.GetMass(),                      # Atomic mass
            atom.GetNumRadicalElectrons(),       # Radical electrons
            atom.GetTotalDegree(),               # Total degree
            atom.GetTotalNumHs(),                # Total hydrogens
            atom.GetTotalValence(),              # Total valence
            int(atom.IsInRing()),                # In ring
            int(atom.IsInRingSize(3)),           # In 3-ring
            int(atom.IsInRingSize(4)),           # In 4-ring
            int(atom.IsInRingSize(5)),           # In 5-ring
            int(atom.IsInRingSize(6)),           # In 6-ring
            int(atom.IsInRingSize(7)),           # In 7-ring
            int(atom.IsInRingSize(8)),           # In 8-ring
        ])
        
        # Additional descriptors
        try:
            features.extend([
                atom.GetAtomicNum(),             # Atomic number
                int(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED),  # Chirality
            ])
        except:
            features.extend([0, 0])  # Default values if properties not available
        
        # Pad to fixed size (64 features total)
        target_size = 64
        while len(features) < target_size:
            features.append(0.0)
        
        return features[:target_size]
    
    def _get_bond_features(self, bond) -> List[float]:
        """Extract comprehensive bond features."""
        bond_type = str(bond.GetBondType())
        bond_idx = self.bond_vocab.get(bond_type, 0)
        
        features = [
            bond_idx,                            # Bond type index
            int(bond.GetIsConjugated()),         # Conjugation
            int(bond.IsInRing()),                # In ring
            bond.GetBondTypeAsDouble(),          # Bond order
        ]
        
        # Stereochemistry
        try:
            stereo = bond.GetStereo()
            features.append(int(stereo))
        except:
            features.append(0)
        
        # Additional bond properties
        try:
            features.extend([
                int(bond.GetIsAromatic()),       # Aromaticity
                bond.GetValenceContrib(bond.GetBeginAtom()),  # Begin valence contribution
                bond.GetValenceContrib(bond.GetEndAtom()),    # End valence contribution
            ])
        except:
            features.extend([0, 0, 0])
        
        # Ensure fixed size (8 features)
        while len(features) < 8:
            features.append(0.0)
            
        return features[:8]
    
    def _compute_graph_properties(self, mol) -> Dict[str, float]:
        """Compute molecular-level graph properties."""
        try:
            from rdkit.Chem import Descriptors
            
            properties = {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'num_rings': rdMolDescriptors.CalcNumRings(mol),
                'num_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
            }
            
            # Additional descriptors
            try:
                properties.update({
                    'tpsa': Descriptors.TPSA(mol),
                    'num_hbd': rdMolDescriptors.CalcNumHBD(mol),
                    'num_hba': rdMolDescriptors.CalcNumHBA(mol),
                    'num_rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
                })
            except:
                pass  # Skip if descriptors not available
            
            return properties
            
        except Exception as e:
            logger.warning(f"Error computing graph properties: {e}")
            return {
                'molecular_weight': 0.0,
                'logp': 0.0,
                'num_rings': 0,
                'num_aromatic_rings': 0,
            }
    
    def _get_empty_graph(self, smiles: str) -> Dict[str, Any]:
        """Return empty graph for invalid SMILES."""
        return {
            'nodes': np.array([[0] * 64], dtype=np.float32),  # Single dummy node
            'edges': np.zeros((2, 0), dtype=np.int64),        # No edges
            'edge_attrs': np.zeros((0, 8), dtype=np.float32), # No edge attributes
            'num_nodes': 1,
            'num_edges': 0,
            'smiles': smiles,
            'molecular_weight': 0.0,
            'num_rings': 0,
            'num_aromatic_rings': 0,
            'logp': 0.0,
            'is_valid': False
        }
    
    def graph_to_pytorch_geometric(self, graph_data: Dict[str, Any]) -> 'torch_geometric.data.Data':
        """
        Convert graph data to PyTorch Geometric format.
        
        Args:
            graph_data: Graph data dictionary
            
        Returns:
            PyTorch Geometric Data object
        """
        try:
            from torch_geometric.data import Data
            
            # Convert to tensors
            x = torch.tensor(graph_data['nodes'], dtype=torch.float)
            edge_index = torch.tensor(graph_data['edges'], dtype=torch.long)
            edge_attr = torch.tensor(graph_data['edge_attrs'], dtype=torch.float)
            
            # Create Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                smiles=graph_data['smiles'],
                molecular_weight=graph_data['molecular_weight'],
                num_rings=graph_data['num_rings'],
                logp=graph_data['logp']
            )
            
            return data
            
        except ImportError:
            logger.warning("PyTorch Geometric not available")
            return None
        except Exception as e:
            logger.error(f"Error converting to PyTorch Geometric: {e}")
            return None
    
    def batch_smiles_to_graphs(self, smiles_list: List[str]) -> List[Dict[str, Any]]:
        """
        Convert multiple SMILES to graph representations efficiently.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            List of graph data dictionaries
        """
        graphs = []
        
        for smiles in smiles_list:
            try:
                graph_data = self.smiles_to_graph(smiles)
                graphs.append(graph_data)
            except Exception as e:
                logger.warning(f"Failed to convert SMILES {smiles}: {e}")
                graphs.append(self._get_empty_graph(smiles))
        
        return graphs
    
    def validate_graph_data(self, graph_data: Dict[str, Any]) -> bool:
        """
        Validate graph data structure.
        
        Args:
            graph_data: Graph data dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_keys = ['nodes', 'edges', 'edge_attrs', 'num_nodes', 'num_edges', 'smiles']
        
        # Check required keys
        for key in required_keys:
            if key not in graph_data:
                logger.error(f"Missing required key: {key}")
                return False
        
        # Check data consistency
        nodes = graph_data['nodes']
        edges = graph_data['edges']
        edge_attrs = graph_data['edge_attrs']
        
        # Validate dimensions
        if len(nodes) != graph_data['num_nodes']:
            logger.error("Node count mismatch")
            return False
        
        if len(edge_attrs) != graph_data['num_edges']:
            logger.error("Edge attribute count mismatch")
            return False
        
        if edges.shape[1] != graph_data['num_edges']:
            logger.error("Edge index count mismatch")
            return False
        
        # Validate edge indices
        if graph_data['num_edges'] > 0:
            max_node_idx = np.max(edges)
            if max_node_idx >= graph_data['num_nodes']:
                logger.error("Invalid edge indices")
                return False
        
        return True


class GraphDataProcessor:
    """Process and manipulate graph data for training."""
    
    def __init__(self, converter: Optional[MolecularGraphConverter] = None):
        """Initialize graph data processor."""
        self.converter = converter or MolecularGraphConverter()
    
    def pad_graphs_to_max_size(self, graphs: List[Dict[str, Any]], 
                              max_nodes: int = 100) -> List[Dict[str, Any]]:
        """
        Pad graphs to uniform size for batch processing.
        
        Args:
            graphs: List of graph data dictionaries
            max_nodes: Maximum number of nodes
            
        Returns:
            List of padded graph data dictionaries
        """
        padded_graphs = []
        
        for graph in graphs:
            padded_graph = self._pad_single_graph(graph, max_nodes)
            padded_graphs.append(padded_graph)
        
        return padded_graphs
    
    def _pad_single_graph(self, graph: Dict[str, Any], max_nodes: int) -> Dict[str, Any]:
        """Pad a single graph to max_nodes size."""
        current_nodes = len(graph['nodes'])
        
        if current_nodes >= max_nodes:
            # Truncate if too large
            padded_graph = {
                'nodes': graph['nodes'][:max_nodes],
                'edges': graph['edges'],
                'edge_attrs': graph['edge_attrs'],
                'num_nodes': max_nodes,
                'num_edges': graph['num_edges'],
                'smiles': graph['smiles'],
                'molecular_weight': graph['molecular_weight'],
                'num_rings': graph['num_rings'],
                'logp': graph['logp'],
                'is_valid': graph.get('is_valid', True)
            }
            
            # Filter edges to keep only those within max_nodes
            valid_edges = []
            valid_edge_attrs = []
            
            for i, edge in enumerate(graph['edges'].T):
                if edge[0] < max_nodes and edge[1] < max_nodes:
                    valid_edges.append(edge)
                    if i < len(graph['edge_attrs']):
                        valid_edge_attrs.append(graph['edge_attrs'][i])
            
            if valid_edges:
                padded_graph['edges'] = np.array(valid_edges).T
                padded_graph['edge_attrs'] = np.array(valid_edge_attrs)
                padded_graph['num_edges'] = len(valid_edge_attrs)
            else:
                padded_graph['edges'] = np.zeros((2, 0), dtype=np.int64)
                padded_graph['edge_attrs'] = np.zeros((0, 8), dtype=np.float32)
                padded_graph['num_edges'] = 0
        
        else:
            # Pad with zeros
            node_feature_size = graph['nodes'].shape[1] if len(graph['nodes']) > 0 else 64
            padding_size = max_nodes - current_nodes
            
            padded_nodes = np.vstack([
                graph['nodes'],
                np.zeros((padding_size, node_feature_size), dtype=np.float32)
            ])
            
            padded_graph = {
                'nodes': padded_nodes,
                'edges': graph['edges'],
                'edge_attrs': graph['edge_attrs'],
                'num_nodes': max_nodes,
                'original_num_nodes': current_nodes,
                'num_edges': graph['num_edges'],
                'smiles': graph['smiles'],
                'molecular_weight': graph['molecular_weight'],
                'num_rings': graph['num_rings'],
                'logp': graph['logp'],
                'is_valid': graph.get('is_valid', True)
            }
        
        return padded_graph
    
    def create_graph_batches(self, graphs: List[Dict[str, Any]], 
                           batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Create batches of graphs for efficient processing.
        
        Args:
            graphs: List of graph data dictionaries
            batch_size: Number of graphs per batch
            
        Returns:
            List of batched graph dictionaries
        """
        batches = []
        
        for i in range(0, len(graphs), batch_size):
            batch_graphs = graphs[i:i + batch_size]
            
            # Find max sizes in batch
            max_nodes = max(g['num_nodes'] for g in batch_graphs)
            max_edges = max(g['num_edges'] for g in batch_graphs)
            
            # Pad all graphs in batch to same size
            padded_batch = self.pad_graphs_to_max_size(batch_graphs, max_nodes)
            
            # Stack into batch tensors
            batch_data = self._stack_graph_batch(padded_batch)
            batches.append(batch_data)
        
        return batches
    
    def _stack_graph_batch(self, graphs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Stack a list of graphs into batch format."""
        batch_nodes = np.stack([g['nodes'] for g in graphs])
        batch_smiles = [g['smiles'] for g in graphs]
        batch_molecular_weights = np.array([g['molecular_weight'] for g in graphs])
        
        # Handle edges separately as they have variable sizes
        batch_edges = [g['edges'] for g in graphs]
        batch_edge_attrs = [g['edge_attrs'] for g in graphs]
        
        return {
            'batch_nodes': batch_nodes,
            'batch_edges': batch_edges,
            'batch_edge_attrs': batch_edge_attrs,
            'batch_smiles': batch_smiles,
            'batch_molecular_weights': batch_molecular_weights,
            'batch_size': len(graphs)
        }