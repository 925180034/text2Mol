"""
Decoders for different output modalities.

This module implements specialized decoders for:
- SMILES generation (handled by MolT5)
- Molecular graph generation  
- Molecular image generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

class SMILESDecoder(nn.Module):
    """
    SMILES decoder (placeholder - actual decoding handled by MolT5)
    This class can be extended for additional SMILES post-processing
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
    def forward(self, features: torch.Tensor, batch: Dict[str, Any], mode: str) -> torch.Tensor:
        """SMILES decoding is handled by MolT5 in core_model.py"""
        return features


class GraphDecoder(nn.Module):
    """
    Molecular graph decoder that generates graph representations
    from fused multi-modal features.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.hidden_size = config.get('hidden_size', 768)
        self.max_atoms = config.get('max_atoms', 50)
        self.num_atom_types = config.get('num_atom_types', 100)  # Atomic numbers
        self.num_bond_types = config.get('num_bond_types', 4)   # Bond types
        
        # Node (atom) generation
        self.node_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size * 2, self.max_atoms * self.num_atom_types)
        )
        
        # Edge (bond) generation
        self.edge_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size * 2, self.max_atoms * self.max_atoms * self.num_bond_types)
        )
        
        # Graph size predictor
        self.size_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.max_atoms),
            nn.Sigmoid()
        )
        
        # Loss functions
        self.node_criterion = nn.CrossEntropyLoss()
        self.edge_criterion = nn.CrossEntropyLoss()
        self.size_criterion = nn.MSELoss()
        
    def forward(self, features: torch.Tensor, batch: Dict[str, Any], mode: str) -> torch.Tensor:
        """
        Generate molecular graph from fused features.
        
        Args:
            features: Fused multi-modal features [batch_size, hidden_size]
            batch: Input batch containing target graphs (for training)
            mode: 'train' or 'inference'
            
        Returns:
            outputs: Graph predictions or generated graphs
        """
        batch_size = features.shape[0]
        
        # Predict graph size (number of atoms)
        size_logits = self.size_predictor(features)  # [batch, max_atoms]
        
        # Predict node types
        node_logits = self.node_predictor(features)  # [batch, max_atoms * num_atom_types]
        node_logits = node_logits.view(batch_size, self.max_atoms, self.num_atom_types)
        
        # Predict edge types
        edge_logits = self.edge_predictor(features)  # [batch, max_atoms * max_atoms * num_bond_types]
        edge_logits = edge_logits.view(batch_size, self.max_atoms, self.max_atoms, self.num_bond_types)
        
        if mode == 'train':
            # Compute training loss
            return self._compute_training_loss(
                node_logits, edge_logits, size_logits, batch
            )
        else:
            # Generate graphs
            return self._generate_graphs(node_logits, edge_logits, size_logits)
    
    def _compute_training_loss(self, node_logits: torch.Tensor, 
                              edge_logits: torch.Tensor,
                              size_logits: torch.Tensor,
                              batch: Dict[str, Any]) -> torch.Tensor:
        """Compute training loss for graph generation"""
        total_loss = 0.0
        
        if 'target_graph' in batch:
            target_graphs = batch['target_graph']
            
            # Extract targets from graph batch
            node_targets, edge_targets, size_targets = self._extract_graph_targets(target_graphs)
            
            # Node prediction loss
            node_loss = self.node_criterion(
                node_logits.reshape(-1, self.num_atom_types),
                node_targets.reshape(-1)
            )
            total_loss += node_loss
            
            # Edge prediction loss  
            edge_loss = self.edge_criterion(
                edge_logits.reshape(-1, self.num_bond_types),
                edge_targets.reshape(-1)
            )
            total_loss += edge_loss
            
            # Size prediction loss
            size_loss = self.size_criterion(size_logits, size_targets)
            total_loss += size_loss * 0.1  # Lower weight for size loss
        
        return total_loss
    
    def _generate_graphs(self, node_logits: torch.Tensor,
                        edge_logits: torch.Tensor, 
                        size_logits: torch.Tensor) -> List[Data]:
        """Generate molecular graphs from predictions"""
        batch_size = node_logits.shape[0]
        generated_graphs = []
        
        for i in range(batch_size):
            # Determine graph size
            size_probs = size_logits[i]  # [max_atoms]
            num_atoms = int(torch.sum(size_probs > 0.5).item())
            num_atoms = max(1, min(num_atoms, self.max_atoms))
            
            # Generate nodes
            node_probs = F.softmax(node_logits[i, :num_atoms], dim=-1)
            node_types = torch.argmax(node_probs, dim=-1)
            
            # Generate edges
            edge_probs = F.softmax(edge_logits[i, :num_atoms, :num_atoms], dim=-1)
            edge_types = torch.argmax(edge_probs, dim=-1)
            
            # Create edge index and edge attributes
            edge_index = []
            edge_attr = []
            
            for j in range(num_atoms):
                for k in range(j + 1, num_atoms):  # Upper triangular
                    bond_type = edge_types[j, k].item()
                    if bond_type > 0:  # 0 means no bond
                        edge_index.extend([[j, k], [k, j]])  # Bidirectional
                        edge_attr.extend([bond_type, bond_type])
            
            if not edge_index:  # Ensure at least one edge
                edge_index = [[0, 0]]
                edge_attr = [1]
            
            # Create PyG Data object
            graph_data = Data(
                x=node_types.float().unsqueeze(-1),
                edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(edge_attr, dtype=torch.long)
            )
            
            generated_graphs.append(graph_data)
        
        return generated_graphs
    
    def _extract_graph_targets(self, graph_batch: Batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract target tensors from graph batch"""
        batch_size = graph_batch.num_graphs
        device = graph_batch.x.device
        
        # Initialize target tensors
        node_targets = torch.zeros(batch_size, self.max_atoms, dtype=torch.long, device=device)
        edge_targets = torch.zeros(batch_size, self.max_atoms, self.max_atoms, dtype=torch.long, device=device)
        size_targets = torch.zeros(batch_size, self.max_atoms, dtype=torch.float, device=device)
        
        # Process each graph in the batch
        ptr = graph_batch.ptr
        for i in range(batch_size):
            start_idx = ptr[i]
            end_idx = ptr[i + 1]
            num_nodes = end_idx - start_idx
            
            if num_nodes > self.max_atoms:
                num_nodes = self.max_atoms
            
            # Node targets
            node_features = graph_batch.x[start_idx:start_idx + num_nodes]
            if node_features.shape[-1] == 1:
                node_targets[i, :num_nodes] = node_features.squeeze(-1).long()
            
            # Size targets
            size_targets[i, :num_nodes] = 1.0
            
            # Edge targets
            edge_mask = (graph_batch.batch[graph_batch.edge_index[0]] == i) & \
                       (graph_batch.batch[graph_batch.edge_index[1]] == i)
            edges = graph_batch.edge_index[:, edge_mask] - start_idx
            edge_attrs = graph_batch.edge_attr[edge_mask] if hasattr(graph_batch, 'edge_attr') else None
            
            for j, (src, dst) in enumerate(edges.t()):
                if src < self.max_atoms and dst < self.max_atoms:
                    bond_type = edge_attrs[j].item() if edge_attrs is not None else 1
                    edge_targets[i, src, dst] = bond_type
        
        return node_targets, edge_targets, size_targets


class ImageDecoder(nn.Module):
    """
    Molecular image decoder that generates 2D molecular structure images
    from fused multi-modal features.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.hidden_size = config.get('hidden_size', 768)
        self.image_size = config.get('image_size', 224)
        self.num_channels = config.get('image_channels', 3)
        
        # Image generation network (similar to GAN generator)
        self.generator = nn.Sequential(
            # Initial projection
            nn.Linear(self.hidden_size, 512 * 4 * 4),
            nn.ReLU(),
            
            # Reshape to feature map
            nn.Unflatten(1, (512, 4, 4)),
            
            # Upsampling layers
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 4x4 -> 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 8x8 -> 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 16x16 -> 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 32x32 -> 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 16, 4, 2, 1),    # 64x64 -> 128x128
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.ConvTranspose2d(16, self.num_channels, 4, 2, 1),  # 128x128 -> 256x256
            nn.Tanh()  # Output in [-1, 1]
        )
        
        # Additional refinement layers
        self.refinement = nn.Sequential(
            nn.Conv2d(self.num_channels, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, self.num_channels, 3, 1, 1),
            nn.Tanh()
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
    def forward(self, features: torch.Tensor, batch: Dict[str, Any], mode: str) -> torch.Tensor:
        """
        Generate molecular images from fused features.
        
        Args:
            features: Fused multi-modal features [batch_size, hidden_size]
            batch: Input batch containing target images (for training)
            mode: 'train' or 'inference'
            
        Returns:
            outputs: Image predictions or generated images
        """
        # Generate base image
        generated_images = self.generator(features)
        
        # Apply refinement
        refined_images = self.refinement(generated_images)
        final_images = generated_images + refined_images  # Residual connection
        
        # Resize to target size if needed
        if final_images.shape[-1] != self.image_size:
            final_images = F.interpolate(
                final_images,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            )
        
        if mode == 'train':
            # Compute training loss
            return self._compute_training_loss(final_images, batch)
        else:
            # Return generated images [batch, channels, height, width]
            return final_images
    
    def _compute_training_loss(self, generated_images: torch.Tensor, 
                              batch: Dict[str, Any]) -> torch.Tensor:
        """Compute training loss for image generation"""
        if 'target_image' in batch:
            target_images = batch['target_image']
            
            # Ensure same size
            if target_images.shape != generated_images.shape:
                target_images = F.interpolate(
                    target_images,
                    size=generated_images.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            # MSE loss
            reconstruction_loss = self.criterion(generated_images, target_images)
            
            # Perceptual loss (simplified - could use VGG features)
            perceptual_loss = self._compute_perceptual_loss(generated_images, target_images)
            
            total_loss = reconstruction_loss + 0.1 * perceptual_loss
            
            return total_loss
        else:
            return torch.tensor(0.0, device=generated_images.device)
    
    def _compute_perceptual_loss(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss using gradient differences"""
        # Compute gradients
        def compute_gradients(images):
            # Sobel filters for edge detection
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                 dtype=torch.float32, device=images.device)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                 dtype=torch.float32, device=images.device)
            
            sobel_x = sobel_x.view(1, 1, 3, 3).repeat(images.shape[1], 1, 1, 1)
            sobel_y = sobel_y.view(1, 1, 3, 3).repeat(images.shape[1], 1, 1, 1)
            
            grad_x = F.conv2d(images, sobel_x, padding=1, groups=images.shape[1])
            grad_y = F.conv2d(images, sobel_y, padding=1, groups=images.shape[1])
            
            return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        
        gen_gradients = compute_gradients(generated)
        target_gradients = compute_gradients(target)
        
        return F.mse_loss(gen_gradients, target_gradients)


class ConditionalDecoder(nn.Module):
    """
    Base class for conditional decoders that can incorporate
    additional conditioning information.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.hidden_size = config.get('hidden_size', 768)
        
        # Conditioning network
        self.condition_encoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
    def forward(self, features: torch.Tensor, 
                condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            features: Main features [batch_size, hidden_size]
            condition: Additional conditioning features [batch_size, hidden_size]
        """
        if condition is not None:
            condition_encoded = self.condition_encoder(condition)
            features = features + condition_encoded
        
        return features


class MultiTaskDecoder(nn.Module):
    """
    Multi-task decoder that can generate multiple output modalities
    simultaneously for multi-task learning.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Individual decoders
        self.smiles_decoder = SMILESDecoder(config)
        self.graph_decoder = GraphDecoder(config) 
        self.image_decoder = ImageDecoder(config)
        
        # Task weighting
        self.task_weights = nn.Parameter(torch.ones(3))  # Learnable task weights
        
    def forward(self, features: torch.Tensor, batch: Dict[str, Any], 
                active_tasks: List[str], mode: str) -> Dict[str, torch.Tensor]:
        """
        Generate outputs for multiple tasks simultaneously.
        
        Args:
            features: Fused features [batch_size, hidden_size]
            batch: Input batch
            active_tasks: List of active task names
            mode: 'train' or 'inference'
            
        Returns:
            outputs: Dictionary of outputs per task
        """
        outputs = {}
        task_losses = {}
        
        task_mapping = {
            'smiles': (self.smiles_decoder, 0),
            'graph': (self.graph_decoder, 1), 
            'image': (self.image_decoder, 2)
        }
        
        for task_name in active_tasks:
            if task_name in task_mapping:
                decoder, task_idx = task_mapping[task_name]
                task_output = decoder(features, batch, mode)
                
                if mode == 'train':
                    # Apply task weighting
                    weighted_output = task_output * torch.softmax(self.task_weights, dim=0)[task_idx]
                    task_losses[task_name] = weighted_output
                else:
                    outputs[task_name] = task_output
        
        if mode == 'train':
            # Return combined loss
            total_loss = sum(task_losses.values())
            return total_loss
        else:
            return outputs