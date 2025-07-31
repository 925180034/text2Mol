"""
Molecular processing and utility functions.

This module provides comprehensive tools for molecular processing,
including SMILES validation, graph/image conversion, and property calculation.
"""

import logging
from typing import Optional, List, Dict, Any, Tuple, Union
import numpy as np
import torch
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from torch_geometric.data import Data
import io
import base64

logger = logging.getLogger(__name__)

class MolecularUtils:
    """Comprehensive molecular utility functions."""
    
    @staticmethod
    def validate_smiles(smiles: str) -> bool:
        """
        Validate SMILES string.
        
        Args:
            smiles: SMILES string to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    @staticmethod
    def canonicalize_smiles(smiles: str, remove_stereo: bool = False) -> Optional[str]:
        """
        Canonicalize SMILES string.
        
        Args:
            smiles: Input SMILES string
            remove_stereo: Whether to remove stereochemistry
            
        Returns:
            Canonical SMILES or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            if remove_stereo:
                Chem.RemoveStereochemistry(mol)
            
            return Chem.MolToSmiles(mol, canonical=True)
        except:
            return None
    
    @staticmethod
    def compute_molecular_properties(smiles: str) -> Dict[str, float]:
        """
        Compute molecular properties from SMILES.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Dictionary of molecular properties
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}
            
            properties = {
                'molecular_weight': rdMolDescriptors.CalcExactMolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'num_atoms': mol.GetNumAtoms(),
                'num_bonds': mol.GetNumBonds(),
                'num_rings': rdMolDescriptors.CalcNumRings(mol),
                'num_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
                'num_aliphatic_rings': rdMolDescriptors.CalcNumAliphaticRings(mol),
                'tpsa': rdMolDescriptors.CalcTPSA(mol),
                'num_hbd': rdMolDescriptors.CalcNumHBD(mol),  # Hydrogen bond donors
                'num_hba': rdMolDescriptors.CalcNumHBA(mol),  # Hydrogen bond acceptors
                'num_rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
                'molar_refractivity': Descriptors.MolMR(mol),
                'fraction_csp3': rdMolDescriptors.CalcFractionCsp3(mol),
                'balaban_j': Descriptors.BalabanJ(mol),
                'bertz_ct': Descriptors.BertzCT(mol)
            }
            
            return properties
            
        except Exception as e:
            logger.warning(f"Error computing properties for {smiles}: {e}")
            return {}
    
    @staticmethod
    def compute_fingerprint(smiles: str, fp_type: str = 'morgan', 
                          **kwargs) -> Optional[np.ndarray]:
        """
        Compute molecular fingerprint.
        
        Args:
            smiles: SMILES string
            fp_type: Fingerprint type ('morgan', 'maccs', 'rdkit', 'avalon')
            **kwargs: Additional parameters for fingerprint computation
            
        Returns:
            Fingerprint as numpy array or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            if fp_type == 'morgan':
                radius = kwargs.get('radius', 2)
                n_bits = kwargs.get('n_bits', 2048)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                
            elif fp_type == 'maccs':
                fp = AllChem.GetMACCSKeysFingerprint(mol)
                
            elif fp_type == 'rdkit':
                n_bits = kwargs.get('n_bits', 2048)
                fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
                
            elif fp_type == 'avalon':
                n_bits = kwargs.get('n_bits', 2048)
                fp = AllChem.GetAvalonFP(mol, nBits=n_bits)
                
            else:
                raise ValueError(f"Unknown fingerprint type: {fp_type}")
            
            # Convert to numpy array
            fp_array = np.zeros((len(fp),), dtype=np.float32)
            for i in range(len(fp)):
                fp_array[i] = fp[i]
            
            return fp_array
            
        except Exception as e:
            logger.warning(f"Error computing {fp_type} fingerprint for {smiles}: {e}")
            return None


def smiles_to_graph(smiles: str, add_self_loops: bool = True) -> Optional[Data]:
    """
    Convert SMILES string to PyTorch Geometric graph.
    
    Args:
        smiles: SMILES string
        add_self_loops: Whether to add self-loops to nodes
        
    Returns:
        PyG Data object or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Add explicit hydrogens for more complete representation
        mol = Chem.AddHs(mol)
        
        # Node features (atoms)
        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),                    # Atomic number
                atom.GetDegree(),                       # Degree
                atom.GetFormalCharge(),                 # Formal charge
                int(atom.GetHybridization()),          # Hybridization
                int(atom.GetIsAromatic()),             # Is aromatic
                atom.GetTotalNumHs(),                   # Total hydrogens
                int(atom.IsInRing()),                   # Is in ring
                int(atom.GetChiralTag()),              # Chirality
                atom.GetMass()                          # Atomic mass
            ]
            atom_features.append(features)
        
        # Edge features (bonds)
        edge_indices = []
        edge_features = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # Add both directions for undirected graph
            edge_indices.extend([[i, j], [j, i]])
            
            # Bond features
            bond_features = [
                int(bond.GetBondType()),               # Bond type
                int(bond.GetIsConjugated()),           # Is conjugated
                int(bond.IsInRing()),                  # Is in ring
                int(bond.GetStereo())                  # Stereochemistry
            ]
            
            edge_features.extend([bond_features, bond_features])
        
        # Add self-loops if requested
        if add_self_loops:
            num_atoms = len(atom_features)
            for i in range(num_atoms):
                edge_indices.append([i, i])
                edge_features.append([0, 0, 0, 0])  # Self-loop features
        
        # Convert to tensors
        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            smiles=smiles,
            num_atoms=len(atom_features)
        )
        
        return data
        
    except Exception as e:
        logger.warning(f"Error converting SMILES to graph: {smiles}, {e}")
        return None


def smiles_to_image(smiles: str, size: Tuple[int, int] = (224, 224),
                   kekulize: bool = True) -> Optional[torch.Tensor]:
    """
    Convert SMILES string to molecular image tensor.
    
    Args:
        smiles: SMILES string
        size: Image size (width, height)
        kekulize: Whether to kekulize the molecule
        
    Returns:
        Image tensor [C, H, W] or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Kekulize if requested
        if kekulize:
            Chem.Kekulize(mol)
        
        # Generate 2D coordinates
        AllChem.Compute2DCoords(mol)
        
        # Draw molecule
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        
        # Get image data
        img_data = drawer.GetDrawingText()
        
        # Convert to PIL Image
        img = Image.open(io.BytesIO(img_data))
        img = img.convert('RGB')
        
        # Convert to tensor
        img_array = np.array(img)
        img_tensor = torch.from_numpy(img_array).float()
        
        # Normalize to [0, 1] and change to CHW format
        img_tensor = img_tensor / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
        
        return img_tensor
        
    except Exception as e:
        logger.warning(f"Error converting SMILES to image: {smiles}, {e}")
        return None


def graph_to_smiles(graph_data: Data, atom_decoder: Optional[Dict[int, str]] = None) -> Optional[str]:
    """
    Convert graph representation back to SMILES.
    
    Args:
        graph_data: PyG Data object
        atom_decoder: Optional mapping from indices to atom symbols
        
    Returns:
        SMILES string or None if conversion fails
    """
    try:
        # Create RDKit molecule
        mol = Chem.RWMol()
        
        # Add atoms
        atom_map = {}
        for i, atom_features in enumerate(graph_data.x):
            if atom_decoder:
                atom_symbol = atom_decoder.get(int(atom_features[0]), 'C')
            else:
                # Use atomic number directly
                atomic_num = int(atom_features[0])
                atom_symbol = Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), atomic_num)
            
            atom = Chem.Atom(atom_symbol)
            atom_idx = mol.AddAtom(atom)
            atom_map[i] = atom_idx
        
        # Add bonds
        edge_index = graph_data.edge_index
        processed_edges = set()
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            
            # Skip self-loops and already processed edges
            if src == dst or (src, dst) in processed_edges or (dst, src) in processed_edges:
                continue
            
            # Get bond type
            if hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None:
                bond_type_idx = int(graph_data.edge_attr[i, 0])
                bond_types = [
                    Chem.BondType.SINGLE,
                    Chem.BondType.DOUBLE,
                    Chem.BondType.TRIPLE,
                    Chem.BondType.AROMATIC
                ]
                bond_type = bond_types[min(bond_type_idx, len(bond_types) - 1)]
            else:
                bond_type = Chem.BondType.SINGLE
            
            mol.AddBond(atom_map[src], atom_map[dst], bond_type)
            processed_edges.add((src, dst))
        
        # Sanitize and get SMILES
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol)
        
        return smiles
        
    except Exception as e:
        logger.warning(f"Error converting graph to SMILES: {e}")
        return None


def image_to_smiles(image_tensor: torch.Tensor, model=None) -> Optional[str]:
    """
    Convert molecular image to SMILES (requires trained OCR model).
    
    Args:
        image_tensor: Image tensor [C, H, W]
        model: Trained image-to-SMILES model
        
    Returns:
        SMILES string or None if conversion fails
    """
    # This would require a trained optical chemical structure recognition model
    # For now, return None as placeholder
    logger.warning("Image to SMILES conversion requires a trained OCR model")
    return None


def compute_tanimoto_similarity(smiles1: str, smiles2: str, 
                               fp_type: str = 'morgan') -> float:
    """
    Compute Tanimoto similarity between two molecules.
    
    Args:
        smiles1: First SMILES string
        smiles2: Second SMILES string
        fp_type: Fingerprint type to use
        
    Returns:
        Tanimoto similarity score (0-1)
    """
    try:
        fp1 = MolecularUtils.compute_fingerprint(smiles1, fp_type)
        fp2 = MolecularUtils.compute_fingerprint(smiles2, fp_type)
        
        if fp1 is None or fp2 is None:
            return 0.0
        
        # Convert to RDKit bit vectors for Tanimoto calculation
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return 0.0
        
        if fp_type == 'morgan':
            fp1_bv = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
            fp2_bv = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        elif fp_type == 'maccs':
            fp1_bv = AllChem.GetMACCSKeysFingerprint(mol1)
            fp2_bv = AllChem.GetMACCSKeysFingerprint(mol2)
        else:
            # Use Morgan as default
            fp1_bv = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
            fp2_bv = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        
        from rdkit import DataStructs
        return DataStructs.TanimotoSimilarity(fp1_bv, fp2_bv)
        
    except Exception as e:
        logger.warning(f"Error computing Tanimoto similarity: {e}")
        return 0.0


def batch_process_smiles(smiles_list: List[str], 
                        operations: List[str] = ['validate', 'canonicalize']) -> Dict[str, List]:
    """
    Batch process a list of SMILES strings.
    
    Args:
        smiles_list: List of SMILES strings
        operations: List of operations to perform
        
    Returns:
        Dictionary with results for each operation
    """
    results = {op: [] for op in operations}
    
    for smiles in smiles_list:
        for operation in operations:
            if operation == 'validate':
                results[operation].append(MolecularUtils.validate_smiles(smiles))
            elif operation == 'canonicalize':
                results[operation].append(MolecularUtils.canonicalize_smiles(smiles))
            elif operation == 'properties':
                results[operation].append(MolecularUtils.compute_molecular_properties(smiles))
            elif operation == 'to_graph':
                results[operation].append(smiles_to_graph(smiles))
            elif operation == 'to_image':
                results[operation].append(smiles_to_image(smiles))
            else:
                logger.warning(f"Unknown operation: {operation}")
                results[operation].append(None)
    
    return results


def save_molecule_image(smiles: str, filename: str, size: Tuple[int, int] = (300, 300)):
    """
    Save molecular structure as image file.
    
    Args:
        smiles: SMILES string
        filename: Output filename
        size: Image size
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        # Generate 2D coordinates
        AllChem.Compute2DCoords(mol)
        
        # Draw and save
        img = Draw.MolToImage(mol, size=size)
        img.save(filename)
        
        logger.info(f"Saved molecule image: {filename}")
        
    except Exception as e:
        logger.error(f"Error saving molecule image: {e}")


def convert_smiles_format(smiles: str, output_format: str) -> Optional[str]:
    """
    Convert SMILES to other molecular formats.
    
    Args:
        smiles: Input SMILES string
        output_format: Target format ('inchi', 'inchi_key', 'mol_block')
        
    Returns:
        Converted string or None if conversion fails
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        if output_format == 'inchi':
            return Chem.MolToInchi(mol)
        elif output_format == 'inchi_key':
            return Chem.MolToInchiKey(mol)
        elif output_format == 'mol_block':
            return Chem.MolToMolBlock(mol)
        else:
            raise ValueError(f"Unknown output format: {output_format}")
            
    except Exception as e:
        logger.warning(f"Error converting SMILES format: {e}")
        return None