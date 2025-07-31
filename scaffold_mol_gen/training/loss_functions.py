"""
Loss functions for scaffold-based molecular generation.

This module implements specialized loss functions for:
- Scaffold preservation
- Multi-modal alignment
- Contrastive learning
- Quality-aware generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
import logging
from rdkit import Chem
from rdkit.Chem import AllChem

from ..utils.mol_utils import MolecularUtils, compute_tanimoto_similarity
from ..utils.scaffold_utils import ScaffoldExtractor

logger = logging.getLogger(__name__)

class ScaffoldLoss(nn.Module):
    """
    Loss function that encourages scaffold preservation in generated molecules.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Loss weights
        self.generation_weight = config.get('generation_weight', 1.0)
        self.scaffold_weight = config.get('scaffold_preservation_weight', 0.5)
        self.validity_weight = config.get('validity_weight', 0.3)
        
        # Base loss functions
        self.generation_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.scaffold_extractor = ScaffoldExtractor()
        
        # Scaffold preservation method
        self.preservation_method = config.get('preservation_method', 'hard')  # 'hard' or 'soft'
        
    def forward(self, outputs: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Compute scaffold-aware loss.
        
        Args:
            outputs: Model outputs (logits for SMILES generation)
            batch: Input batch containing targets and scaffold information
            
        Returns:
            Combined loss tensor
        """
        total_loss = 0.0
        
        # 1. Generation loss (standard cross-entropy)
        if 'target_tokens' in batch:
            target_ids = batch['target_tokens']['input_ids']
            
            # Reshape for loss computation
            if outputs.dim() == 3:  # [batch, seq_len, vocab_size]
                vocab_size = outputs.size(-1)
                flat_outputs = outputs.view(-1, vocab_size)
                flat_targets = target_ids.view(-1)
            else:
                flat_outputs = outputs
                flat_targets = target_ids
            
            generation_loss = self.generation_loss(flat_outputs, flat_targets)
            total_loss += self.generation_weight * generation_loss
        
        # 2. Scaffold preservation loss
        if self.scaffold_weight > 0 and 'raw_data' in batch:
            scaffold_loss = self._compute_scaffold_loss(outputs, batch)
            total_loss += self.scaffold_weight * scaffold_loss
        
        # 3. Validity loss
        if self.validity_weight > 0 and 'raw_data' in batch:
            validity_loss = self._compute_validity_loss(outputs, batch)
            total_loss += self.validity_weight * validity_loss
        
        return total_loss
    
    def _compute_scaffold_loss(self, outputs: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute loss that encourages scaffold preservation."""
        # This requires decoding outputs to SMILES and comparing scaffolds
        # For efficiency during training, we use a simplified approach
        
        if self.preservation_method == 'hard':
            # Hard constraint: exact scaffold match required
            return self._hard_scaffold_loss(outputs, batch)
        else:
            # Soft constraint: encourage scaffold similarity
            return self._soft_scaffold_loss(outputs, batch)
    
    def _hard_scaffold_loss(self, outputs: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        """Hard scaffold preservation constraint."""
        # During training, we use a proxy loss based on scaffold tokens
        if 'scaffold_tokens' in batch and 'target_tokens' in batch:
            scaffold_ids = batch['scaffold_tokens']['input_ids']
            target_ids = batch['target_tokens']['input_ids']
            
            # Create mask for scaffold positions in target
            # This is a simplified approach - in practice, you'd need more sophisticated
            # alignment between scaffold and full molecule tokens
            
            # For now, return a placeholder loss
            return torch.tensor(0.0, device=outputs.device, requires_grad=True)
        
        return torch.tensor(0.0, device=outputs.device)
    
    def _soft_scaffold_loss(self, outputs: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        """Soft scaffold preservation encouragement."""
        # Placeholder implementation
        # In practice, this would involve:
        # 1. Sampling from the output distribution
        # 2. Decoding to SMILES
        # 3. Computing scaffold similarity
        # 4. Using similarity as loss weight
        
        return torch.tensor(0.0, device=outputs.device)
    
    def _compute_validity_loss(self, outputs: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute loss that encourages valid molecule generation."""
        # This is challenging to implement during training without expensive decoding
        # A proxy approach is to use learned validity prediction
        
        # Placeholder - could be implemented with a discriminator
        return torch.tensor(0.0, device=outputs.device)


class MultiModalLoss(nn.Module):
    """
    Loss function for multi-modal molecular generation.
    Handles alignment between different modalities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Task-specific loss weights
        self.task_weights = config.get('task_weights', {
            'smiles': 1.0,
            'graph': 0.5,
            'image': 0.3
        })
        
        # Alignment loss weight
        self.alignment_weight = config.get('alignment_weight', 0.2)
        
        # Individual loss functions
        self.smiles_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.graph_loss = nn.MSELoss()
        self.image_loss = nn.MSELoss()
        
    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> torch.Tensor:
        """
        Compute multi-modal loss.
        
        Args:
            outputs: Dictionary of outputs for each modality
            batch: Input batch
            
        Returns:
            Combined multi-modal loss
        """
        total_loss = 0.0
        
        # Task-specific losses
        for task, task_outputs in outputs.items():
            if task == 'smiles' and 'target_tokens' in batch:
                target_ids = batch['target_tokens']['input_ids']
                if task_outputs.dim() == 3:
                    vocab_size = task_outputs.size(-1)
                    flat_outputs = task_outputs.view(-1, vocab_size)
                    flat_targets = target_ids.view(-1)
                else:
                    flat_outputs = task_outputs
                    flat_targets = target_ids
                
                task_loss = self.smiles_loss(flat_outputs, flat_targets)
                total_loss += self.task_weights.get(task, 1.0) * task_loss
                
            elif task == 'graph' and 'target_graph' in batch:
                # Graph generation loss
                task_loss = self.graph_loss(task_outputs, batch['target_graph'])
                total_loss += self.task_weights.get(task, 1.0) * task_loss
                
            elif task == 'image' and 'target_image' in batch:
                # Image generation loss
                task_loss = self.image_loss(task_outputs, batch['target_image'])
                total_loss += self.task_weights.get(task, 1.0) * task_loss
        
        # Modal alignment loss
        if self.alignment_weight > 0 and len(outputs) > 1:
            alignment_loss = self._compute_alignment_loss(outputs)
            total_loss += self.alignment_weight * alignment_loss
        
        return total_loss
    
    def _compute_alignment_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute alignment loss between different modalities."""
        # Compute pairwise alignment losses
        alignment_loss = 0.0
        pairs = 0
        
        output_items = list(outputs.items())
        for i in range(len(output_items)):
            for j in range(i + 1, len(output_items)):
                task1, out1 = output_items[i]
                task2, out2 = output_items[j]
                
                # Flatten outputs to same dimensionality
                out1_flat = self._flatten_output(out1)
                out2_flat = self._flatten_output(out2)
                
                # Compute cosine similarity
                similarity = F.cosine_similarity(out1_flat, out2_flat, dim=-1)
                
                # Encourage high similarity (minimize negative similarity)
                alignment_loss += -similarity.mean()
                pairs += 1
        
        return alignment_loss / pairs if pairs > 0 else torch.tensor(0.0)
    
    def _flatten_output(self, output: torch.Tensor) -> torch.Tensor:
        """Flatten output to consistent shape for alignment."""
        if output.dim() > 2:
            return output.view(output.size(0), -1)
        return output


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning better molecular representations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        self.temperature = config.get('temperature', 0.07)
        self.margin = config.get('margin', 0.5)
        self.similarity_metric = config.get('similarity_metric', 'cosine')  # 'cosine', 'euclidean'
        
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            anchor: Anchor representations [batch_size, dim]
            positive: Positive (similar) representations [batch_size, dim]
            negative: Negative (dissimilar) representations [batch_size, dim]
            
        Returns:
            Contrastive loss
        """
        if self.similarity_metric == 'cosine':
            return self._cosine_contrastive_loss(anchor, positive, negative)
        else:
            return self._euclidean_contrastive_loss(anchor, positive, negative)
    
    def _cosine_contrastive_loss(self, anchor: torch.Tensor, positive: torch.Tensor,
                                negative: torch.Tensor) -> torch.Tensor:
        """Cosine similarity based contrastive loss."""
        # Normalize embeddings
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negative = F.normalize(negative, dim=-1)
        
        # Compute similarities
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature
        neg_sim = torch.sum(anchor * negative, dim=-1) / self.temperature
        
        # Contrastive loss
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim.unsqueeze(-1)], dim=-1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        return F.cross_entropy(logits, labels)
    
    def _euclidean_contrastive_loss(self, anchor: torch.Tensor, positive: torch.Tensor,
                                  negative: torch.Tensor) -> torch.Tensor:
        """Euclidean distance based contrastive loss."""
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        
        # Margin-based contrastive loss
        loss = torch.mean(pos_dist + F.relu(self.margin - neg_dist))
        
        return loss


class QualityAwareLoss(nn.Module):
    """
    Loss function that incorporates molecular quality metrics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Quality weights
        self.generation_weight = config.get('generation_weight', 1.0)
        self.quality_weight = config.get('quality_weight', 0.3)
        
        # Quality metrics to optimize
        self.quality_metrics = config.get('quality_metrics', [
            'validity', 'uniqueness', 'novelty', 'drug_likeness'
        ])
        
        # Base generation loss
        self.generation_loss = nn.CrossEntropyLoss(ignore_index=-100)
        
    def forward(self, outputs: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Compute quality-aware loss.
        
        Args:
            outputs: Model outputs
            batch: Input batch
            
        Returns:
            Quality-aware loss
        """
        total_loss = 0.0
        
        # Standard generation loss
        if 'target_tokens' in batch:
            target_ids = batch['target_tokens']['input_ids']
            
            if outputs.dim() == 3:
                vocab_size = outputs.size(-1)
                flat_outputs = outputs.view(-1, vocab_size)
                flat_targets = target_ids.view(-1)
            else:
                flat_outputs = outputs
                flat_targets = target_ids
            
            generation_loss = self.generation_loss(flat_outputs, flat_targets)
            total_loss += self.generation_weight * generation_loss
        
        # Quality-based loss (requires sampling and evaluation)
        if self.quality_weight > 0:
            quality_loss = self._compute_quality_loss(outputs, batch)
            total_loss += self.quality_weight * quality_loss
        
        return total_loss
    
    def _compute_quality_loss(self, outputs: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute quality-based loss."""
        # This is computationally expensive during training
        # In practice, you might use a separate quality predictor network
        # or apply this loss only occasionally
        
        # Placeholder implementation
        return torch.tensor(0.0, device=outputs.device)


class AdversarialLoss(nn.Module):
    """
    Adversarial loss for training with discriminator.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        self.generator_weight = config.get('generator_weight', 1.0)
        self.discriminator_weight = config.get('discriminator_weight', 0.5)
        
        # Loss type
        self.loss_type = config.get('loss_type', 'standard')  # 'standard', 'wgan', 'lsgan'
        
    def generator_loss(self, discriminator_outputs: torch.Tensor) -> torch.Tensor:
        """Compute generator loss."""
        if self.loss_type == 'standard':
            # Standard GAN loss: log(1 - D(G(z)))
            return -torch.mean(torch.log(discriminator_outputs + 1e-8))
        elif self.loss_type == 'wgan':
            # WGAN loss: -D(G(z))
            return -torch.mean(discriminator_outputs)
        elif self.loss_type == 'lsgan':
            # LSGAN loss: (D(G(z)) - 1)^2
            return torch.mean((discriminator_outputs - 1) ** 2)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def discriminator_loss(self, real_outputs: torch.Tensor, 
                          fake_outputs: torch.Tensor) -> torch.Tensor:
        """Compute discriminator loss."""
        if self.loss_type == 'standard':
            # Standard GAN loss
            real_loss = -torch.mean(torch.log(real_outputs + 1e-8))
            fake_loss = -torch.mean(torch.log(1 - fake_outputs + 1e-8))
            return real_loss + fake_loss
        elif self.loss_type == 'wgan':
            # WGAN loss
            return torch.mean(fake_outputs) - torch.mean(real_outputs)
        elif self.loss_type == 'lsgan':
            # LSGAN loss
            real_loss = torch.mean((real_outputs - 1) ** 2)
            fake_loss = torch.mean(fake_outputs ** 2)
            return real_loss + fake_loss
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class CombinedLoss(nn.Module):
    """
    Combined loss function that integrates multiple loss components.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Initialize component losses
        self.losses = {}
        
        if config.get('use_scaffold_loss', True):
            self.losses['scaffold'] = ScaffoldLoss(config.get('scaffold_loss', {}))
        
        if config.get('use_multimodal_loss', False):
            self.losses['multimodal'] = MultiModalLoss(config.get('multimodal_loss', {}))
        
        if config.get('use_contrastive_loss', False):
            self.losses['contrastive'] = ContrastiveLoss(config.get('contrastive_loss', {}))
        
        if config.get('use_quality_loss', False):
            self.losses['quality'] = QualityAwareLoss(config.get('quality_loss', {}))
        
        # Loss combination weights
        self.loss_weights = config.get('loss_weights', {
            'scaffold': 1.0,
            'multimodal': 0.5,
            'contrastive': 0.3,
            'quality': 0.2
        })
        
    def forward(self, outputs: Any, batch: Dict[str, Any], 
                additional_inputs: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            outputs: Model outputs
            batch: Input batch
            additional_inputs: Additional inputs for specific loss components
            
        Returns:
            Dictionary of individual and total losses
        """
        losses = {}
        total_loss = 0.0
        
        for loss_name, loss_fn in self.losses.items():
            try:
                if loss_name == 'contrastive' and additional_inputs:
                    # Contrastive loss needs special inputs
                    anchor = additional_inputs.get('anchor')
                    positive = additional_inputs.get('positive')
                    negative = additional_inputs.get('negative')
                    
                    if anchor is not None and positive is not None and negative is not None:
                        loss_value = loss_fn(anchor, positive, negative)
                    else:
                        loss_value = torch.tensor(0.0, device=outputs.device)
                else:
                    loss_value = loss_fn(outputs, batch)
                
                weight = self.loss_weights.get(loss_name, 1.0)
                weighted_loss = weight * loss_value
                
                losses[loss_name] = loss_value
                losses[f'{loss_name}_weighted'] = weighted_loss
                total_loss += weighted_loss
                
            except Exception as e:
                logger.warning(f"Error computing {loss_name} loss: {e}")
                losses[loss_name] = torch.tensor(0.0)
        
        losses['total'] = total_loss
        return losses