"""
Scaffold-Based Multi-Modal Molecular Generation

A state-of-the-art system for generating molecules while preserving 
molecular scaffolds across multiple modalities.
"""

__version__ = "1.0.0-alpha"
__author__ = "Scaffold-Mol-Gen Team"
__email__ = "contact@scaffold-mol-gen.com"

from .models.core_model import ScaffoldBasedMolT5Generator
from .api.interactive import InteractiveMoleculeDesigner
from .utils.scaffold_utils import ScaffoldExtractor
from .evaluation.evaluator import ModelEvaluator

__all__ = [
    "ScaffoldBasedMolT5Generator",
    "InteractiveMoleculeDesigner", 
    "ScaffoldExtractor",
    "ModelEvaluator"
]