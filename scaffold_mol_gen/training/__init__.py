"""
Training modules for scaffold-based molecular generation.

This package contains:
- Training orchestration and management
- Loss functions and optimization
- Evaluation metrics and validation
- Checkpointing and model persistence
"""

from .trainer import ScaffoldMolTrainer, MultiTaskTrainer
from .loss_functions import ScaffoldLoss, MultiModalLoss, ContrastiveLoss
from .metrics import MolecularMetrics, GenerationMetrics
from .optimizer import get_optimizer, get_scheduler

__all__ = [
    'ScaffoldMolTrainer',
    'MultiTaskTrainer',
    'ScaffoldLoss',
    'MultiModalLoss', 
    'ContrastiveLoss',
    'MolecularMetrics',
    'GenerationMetrics',
    'get_optimizer',
    'get_scheduler'
]