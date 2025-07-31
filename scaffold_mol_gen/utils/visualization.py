"""
Molecular visualization utilities.

This module provides tools for visualizing molecules, scaffolds, and
generation results with various plotting and analysis capabilities.
"""

import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any, Union
from PIL import Image
import torch
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
import io
import base64

logger = logging.getLogger(__name__)

class MolecularVisualizer:
    """Comprehensive molecular visualization toolkit."""
    
    def __init__(self, style: str = 'default'):
        """
        Initialize visualizer with style settings.
        
        Args:
            style: Visualization style ('default', 'publication', 'presentation')
        """
        self.style = style
        self._setup_style()
        
    def _setup_style(self):
        """Setup visualization style parameters."""
        if self.style == 'publication':
            plt.style.use('seaborn-v0_8-whitegrid')
            self.figsize = (12, 8)
            self.dpi = 300
        elif self.style == 'presentation':
            plt.style.use('seaborn-v0_8-talk')
            self.figsize = (16, 10)
            self.dpi = 150
        else:
            plt.style.use('default')
            self.figsize = (10, 6)
            self.dpi = 100
    
    def draw_molecule(self, smiles: str, size: Tuple[int, int] = (300, 300),
                     highlight_atoms: Optional[List[int]] = None,
                     highlight_bonds: Optional[List[int]] = None) -> Image.Image:
        """
        Draw molecular structure from SMILES.
        
        Args:
            smiles: SMILES string
            size: Image size (width, height)
            highlight_atoms: Atom indices to highlight
            highlight_bonds: Bond indices to highlight
            
        Returns:
            PIL Image of the molecule
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            
            # Generate 2D coordinates
            AllChem.Compute2DCoords(mol)
            
            # Set up drawing options
            draw_options = rdMolDraw2D.MolDrawOptions()
            draw_options.addStereoAnnotation = True
            draw_options.addAtomIndices = False
            
            # Create drawer
            drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
            drawer.SetDrawOptions(draw_options)
            
            # Highlight if specified
            if highlight_atoms or highlight_bonds:
                highlight_atom_colors = {}
                highlight_bond_colors = {}
                
                if highlight_atoms:
                    for atom_idx in highlight_atoms:
                        highlight_atom_colors[atom_idx] = (1, 0, 0)  # Red
                        
                if highlight_bonds:
                    for bond_idx in highlight_bonds:
                        highlight_bond_colors[bond_idx] = (1, 0, 0)  # Red
                
                drawer.DrawMolecule(
                    mol,
                    highlightAtoms=highlight_atoms or [],
                    highlightAtomColors=highlight_atom_colors,
                    highlightBonds=highlight_bonds or [],
                    highlightBondColors=highlight_bond_colors
                )
            else:
                drawer.DrawMolecule(mol)
            
            drawer.FinishDrawing()
            
            # Convert to PIL Image
            img_data = drawer.GetDrawingText()
            img = Image.open(io.BytesIO(img_data))
            
            return img
            
        except Exception as e:
            logger.error(f"Error drawing molecule {smiles}: {e}")
            # Return blank image
            return Image.new('RGB', size, 'white')
    
    def plot_scaffold_diversity(self, scaffolds: List[str], 
                               title: str = "Scaffold Diversity Analysis") -> plt.Figure:
        """
        Plot scaffold diversity statistics.
        
        Args:
            scaffolds: List of scaffold SMILES
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        from .scaffold_utils import ScaffoldExtractor
        
        # Compute statistics
        stats = ScaffoldExtractor.scaffold_statistics(scaffolds)
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        fig.suptitle(title, fontsize=16)
        
        # Scaffold count distribution
        scaffold_counts = list(stats['scaffold_counts'].values())
        axes[0, 0].hist(scaffold_counts, bins=30, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Scaffold Frequency Distribution')
        axes[0, 0].set_xlabel('Frequency')
        axes[0, 0].set_ylabel('Number of Scaffolds')
        
        # Scaffold size distribution
        scaffold_sizes = []
        for scaffold in stats['scaffold_counts'].keys():
            mol = Chem.MolFromSmiles(scaffold)
            if mol:
                scaffold_sizes.append(mol.GetNumAtoms())
        
        axes[0, 1].hist(scaffold_sizes, bins=20, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('Scaffold Size Distribution')
        axes[0, 1].set_xlabel('Number of Atoms')
        axes[0, 1].set_ylabel('Number of Scaffolds')
        
        # Top 10 most common scaffolds
        top_scaffolds = sorted(stats['scaffold_counts'].items(), 
                              key=lambda x: x[1], reverse=True)[:10]
        scaffold_names = [f"S{i+1}" for i in range(len(top_scaffolds))]
        counts = [count for _, count in top_scaffolds]
        
        axes[1, 0].bar(scaffold_names, counts, color='coral')
        axes[1, 0].set_title('Top 10 Most Common Scaffolds')
        axes[1, 0].set_xlabel('Scaffold')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Summary statistics
        summary_text = f"""
        Total Molecules: {stats['total_molecules']}
        Unique Scaffolds: {stats['unique_scaffolds']}
        Diversity: {stats['scaffold_diversity']:.3f}
        Avg Size: {stats['avg_scaffold_size']:.1f}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, 
                        verticalalignment='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Summary Statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_molecular_properties(self, smiles_list: List[str],
                                 properties: Optional[List[str]] = None,
                                 title: str = "Molecular Properties Distribution") -> plt.Figure:
        """
        Plot distribution of molecular properties.
        
        Args:
            smiles_list: List of SMILES strings
            properties: Properties to plot (default: common drug-like properties)
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        from .mol_utils import MolecularUtils
        
        if properties is None:
            properties = ['molecular_weight', 'logp', 'tpsa', 'num_hbd', 'num_hba']
        
        # Compute properties for all molecules
        all_properties = []
        for smiles in smiles_list:
            props = MolecularUtils.compute_molecular_properties(smiles)
            if props:
                all_properties.append(props)
        
        if not all_properties:
            logger.warning("No valid molecules found")
            return plt.figure()
        
        # Create DataFrame
        df = pd.DataFrame(all_properties)
        
        # Plot
        n_props = len(properties)
        n_cols = min(3, n_props)
        n_rows = (n_props + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=self.figsize, dpi=self.dpi)
        fig.suptitle(title, fontsize=16)
        
        if n_props == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 else axes
        
        for i, prop in enumerate(properties):
            if prop in df.columns:
                df[prop].hist(bins=30, alpha=0.7, ax=axes[i])
                axes[i].set_title(f'{prop.replace("_", " ").title()}')
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Frequency')
            else:
                axes[i].text(0.5, 0.5, f'Property\n{prop}\nnot found', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(prop)
        
        # Hide unused subplots
        for i in range(n_props, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_generation_results(self, original_molecules: List[str],
                               generated_molecules: List[str],
                               scaffolds: List[str],
                               title: str = "Generation Results") -> plt.Figure:
        """
        Plot comparison between original and generated molecules.
        
        Args:
            original_molecules: Original SMILES strings
            generated_molecules: Generated SMILES strings
            scaffolds: Scaffold SMILES strings
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        from .mol_utils import MolecularUtils, compute_tanimoto_similarity
        
        # Compute similarities
        similarities = []
        for orig, gen in zip(original_molecules, generated_molecules):
            sim = compute_tanimoto_similarity(orig, gen)
            similarities.append(sim)
        
        # Compute validity
        orig_valid = [MolecularUtils.validate_smiles(s) for s in original_molecules]
        gen_valid = [MolecularUtils.validate_smiles(s) for s in generated_molecules]
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        fig.suptitle(title, fontsize=16)
        
        # Similarity distribution
        axes[0, 0].hist(similarities, bins=20, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Tanimoto Similarity Distribution')
        axes[0, 0].set_xlabel('Similarity')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(similarities), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(similarities):.3f}')
        axes[0, 0].legend()
        
        # Validity comparison
        validity_data = {
            'Original': [sum(orig_valid), len(orig_valid) - sum(orig_valid)],
            'Generated': [sum(gen_valid), len(gen_valid) - sum(gen_valid)]
        }
        
        x = np.arange(2)
        width = 0.35
        axes[0, 1].bar(x - width/2, validity_data['Original'], width, 
                      label='Original', alpha=0.7, color='green')
        axes[0, 1].bar(x + width/2, validity_data['Generated'], width,
                      label='Generated', alpha=0.7, color='orange')
        axes[0, 1].set_title('Validity Comparison')
        axes[0, 1].set_xlabel('Validity')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(['Valid', 'Invalid'])
        axes[0, 1].legend()
        
        # Scaffold preservation
        scaffold_preserved = []
        for scaffold, gen in zip(scaffolds, generated_molecules):
            if MolecularUtils.validate_smiles(gen):
                gen_mol = Chem.MolFromSmiles(gen)
                scaffold_mol = Chem.MolFromSmiles(scaffold)
                if gen_mol and scaffold_mol:
                    preserved = gen_mol.HasSubstructMatch(scaffold_mol)
                    scaffold_preserved.append(preserved)
                else:
                    scaffold_preserved.append(False)
            else:
                scaffold_preserved.append(False)
        
        preservation_rate = sum(scaffold_preserved) / len(scaffold_preserved) if scaffold_preserved else 0
        
        axes[1, 0].pie([sum(scaffold_preserved), len(scaffold_preserved) - sum(scaffold_preserved)],
                      labels=['Preserved', 'Not Preserved'], autopct='%1.1f%%',
                      colors=['lightgreen', 'lightcoral'])
        axes[1, 0].set_title(f'Scaffold Preservation\n({preservation_rate:.1%})')
        
        # Summary statistics
        summary_text = f"""
        Total Molecules: {len(generated_molecules)}
        Valid Generated: {sum(gen_valid)} ({sum(gen_valid)/len(gen_valid):.1%})
        Mean Similarity: {np.mean(similarities):.3f}
        Scaffold Preserved: {sum(scaffold_preserved)} ({preservation_rate:.1%})
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, 
                        verticalalignment='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Summary Statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_molecule_grid(self, smiles_list: List[str], 
                           titles: Optional[List[str]] = None,
                           mols_per_row: int = 4,
                           mol_size: Tuple[int, int] = (200, 200)) -> Image.Image:
        """
        Create a grid of molecular structures.
        
        Args:
            smiles_list: List of SMILES strings
            titles: Optional titles for each molecule
            mols_per_row: Number of molecules per row
            mol_size: Size of each molecule image
            
        Returns:
            PIL Image containing the grid
        """
        n_mols = len(smiles_list)
        n_rows = (n_mols + mols_per_row - 1) // mols_per_row
        
        # Create individual molecule images
        mol_images = []
        for i, smiles in enumerate(smiles_list):
            img = self.draw_molecule(smiles, mol_size)
            
            # Add title if provided
            if titles and i < len(titles):
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(img)
                # Try to use a font, fallback to default if not available
                try:
                    font = ImageFont.truetype("arial.ttf", 14)
                except:
                    font = ImageFont.load_default()
                
                # Draw title at the bottom
                text_bbox = draw.textbbox((0, 0), titles[i], font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_x = (mol_size[0] - text_width) // 2
                draw.text((text_x, mol_size[1] - 20), titles[i], fill='black', font=font)
            
            mol_images.append(img)
        
        # Create grid
        grid_width = mols_per_row * mol_size[0]
        grid_height = n_rows * mol_size[1]
        grid_image = Image.new('RGB', (grid_width, grid_height), 'white')
        
        for i, img in enumerate(mol_images):
            row = i // mols_per_row
            col = i % mols_per_row
            x = col * mol_size[0]
            y = row * mol_size[1]
            grid_image.paste(img, (x, y))
        
        return grid_image
    
    def plot_training_metrics(self, metrics: Dict[str, List[float]],
                            title: str = "Training Metrics") -> plt.Figure:
        """
        Plot training metrics over time.
        
        Args:
            metrics: Dictionary of metric names to values over time
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        n_metrics = len(metrics)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=self.figsize, dpi=self.dpi)
        fig.suptitle(title, fontsize=16)
        
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_metrics))
        
        for i, (metric_name, values) in enumerate(metrics.items()):
            epochs = range(1, len(values) + 1)
            axes[i].plot(epochs, values, color=colors[i], linewidth=2, marker='o')
            axes[i].set_title(metric_name.replace('_', ' ').title())
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel('Value')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    def save_visualization(self, fig: plt.Figure, filename: str, 
                          format: str = 'png', dpi: Optional[int] = None):
        """
        Save visualization to file.
        
        Args:
            fig: Matplotlib figure
            filename: Output filename
            format: File format ('png', 'pdf', 'svg')
            dpi: Resolution (uses instance default if None)
        """
        if dpi is None:
            dpi = self.dpi
            
        fig.savefig(filename, format=format, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved visualization: {filename}")
    
    def interactive_molecule_explorer(self, smiles_list: List[str],
                                    properties: Optional[List[str]] = None):
        """
        Create interactive molecule explorer (requires notebook environment).
        
        Args:
            smiles_list: List of SMILES strings
            properties: Properties to display
        """
        try:
            import ipywidgets as widgets
            from IPython.display import display, HTML
            
            if properties is None:
                properties = ['molecular_weight', 'logp', 'tpsa']
            
            def show_molecule(idx):
                if 0 <= idx < len(smiles_list):
                    smiles = smiles_list[idx]
                    img = self.draw_molecule(smiles)
                    
                    # Display molecule
                    display(img)
                    
                    # Display properties
                    from .mol_utils import MolecularUtils
                    props = MolecularUtils.compute_molecular_properties(smiles)
                    
                    prop_html = f"<h3>Molecule {idx + 1}</h3>"
                    prop_html += f"<p><strong>SMILES:</strong> {smiles}</p>"
                    
                    for prop in properties:
                        if prop in props:
                            prop_html += f"<p><strong>{prop.replace('_', ' ').title()}:</strong> {props[prop]:.3f}</p>"
                    
                    display(HTML(prop_html))
            
            # Create slider
            slider = widgets.IntSlider(
                value=0,
                min=0,
                max=len(smiles_list) - 1,
                step=1,
                description='Molecule:',
                disabled=False,
                continuous_update=False,
                orientation='horizontal',
                readout=True,
                readout_format='d'
            )
            
            widgets.interact(show_molecule, idx=slider)
            
        except ImportError:
            logger.warning("Interactive explorer requires jupyter widgets")