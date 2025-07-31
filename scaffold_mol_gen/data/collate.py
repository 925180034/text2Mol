"""
Collate functions for multi-modal molecular data.

This module provides utilities for batching and collating
different types of molecular data (text, SMILES, graphs, images).
"""

import torch
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data, Batch
from typing import List, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

def multimodal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for multi-modal molecular data.
    
    Handles batching of:
    - Text tokens (padded sequences)
    - SMILES tokens (padded sequences) 
    - Molecular graphs (PyG Batch)
    - Molecular images (stacked tensors)
    - Raw data (lists)
    
    Args:
        batch: List of sample dictionaries from dataset
        
    Returns:
        Collated batch dictionary
    """
    if not batch:
        return {}
    
    collated = {}
    
    # Get all keys from the first non-empty sample
    sample_keys = set()
    for sample in batch:
        if sample:
            sample_keys.update(sample.keys())
    
    for key in sample_keys:
        values = [sample.get(key) for sample in batch if sample.get(key) is not None]
        
        if not values:
            continue
        
        try:
            if key in ['text_tokens', 'smiles_tokens', 'scaffold_tokens', 'target_tokens']:
                collated[key] = collate_tokens(values)
            elif key in ['scaffold_graph', 'additional_graph', 'target_graph']:
                collated[key] = collate_graphs(values)
            elif key in ['scaffold_image', 'additional_image', 'target_image']:
                collated[key] = collate_images(values)
            elif key == 'raw_data':
                collated[key] = collate_raw_data(values)
            elif key == 'index':
                collated[key] = torch.tensor(values, dtype=torch.long)
            else:
                # Try to handle other types
                collated[key] = collate_generic(values)
                
        except Exception as e:
            logger.warning(f"Error collating key '{key}': {e}")
            # Skip problematic keys rather than failing
            continue
    
    return collated


def collate_tokens(token_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate tokenized sequences with padding.
    
    Args:
        token_list: List of token dictionaries with 'input_ids' and 'attention_mask'
        
    Returns:
        Batched token dictionary
    """
    if not token_list:
        return {}
    
    # Extract input_ids and attention_masks
    input_ids = [tokens['input_ids'] for tokens in token_list if 'input_ids' in tokens]
    attention_masks = [tokens['attention_mask'] for tokens in token_list if 'attention_mask' in tokens]
    
    if not input_ids:
        return {}
    
    # Stack tensors (assuming they're already padded to same length)
    try:
        batched_input_ids = torch.stack(input_ids, dim=0)
        batched_attention_masks = torch.stack(attention_masks, dim=0) if attention_masks else None
        
        result = {'input_ids': batched_input_ids}
        if batched_attention_masks is not None:
            result['attention_mask'] = batched_attention_masks
        
        return result
        
    except Exception as e:
        logger.warning(f"Error stacking tokens, trying padding: {e}")
        
        # If stacking fails, try padding
        return pad_token_sequences(input_ids, attention_masks)


def pad_token_sequences(input_ids: List[torch.Tensor], 
                       attention_masks: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Pad token sequences to same length.
    
    Args:
        input_ids: List of input ID tensors
        attention_masks: List of attention mask tensors
        
    Returns:
        Padded and batched tensors
    """
    # Pad sequences
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    
    result = {'input_ids': padded_input_ids}
    
    if attention_masks:
        padded_attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        result['attention_mask'] = padded_attention_masks
    
    return result


def collate_graphs(graph_list: List[Data]) -> Batch:
    """
    Collate molecular graphs using PyTorch Geometric batching.
    
    Args:
        graph_list: List of PyG Data objects
        
    Returns:
        Batched graph object
    """
    if not graph_list:
        return Batch()
    
    try:
        # Filter out None values
        valid_graphs = [g for g in graph_list if g is not None]
        
        if not valid_graphs:
            return Batch()
        
        # Use PyG's batching
        return Batch.from_data_list(valid_graphs)
        
    except Exception as e:
        logger.error(f"Error batching graphs: {e}")
        return Batch()


def collate_images(image_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Collate molecular images.
    
    Args:
        image_list: List of image tensors [C, H, W]
        
    Returns:
        Batched image tensor [B, C, H, W]
    """
    if not image_list:
        return torch.empty(0)
    
    try:
        # Filter out None values
        valid_images = [img for img in image_list if img is not None]
        
        if not valid_images:
            return torch.empty(0)
        
        # Stack images
        return torch.stack(valid_images, dim=0)
        
    except Exception as e:
        logger.error(f"Error batching images: {e}")
        # Return empty tensor with correct dimensions
        if valid_images:
            sample_shape = valid_images[0].shape
            return torch.empty(0, *sample_shape)
        return torch.empty(0)


def collate_raw_data(raw_list: List[Dict[str, str]]) -> Dict[str, List[str]]:
    """
    Collate raw data (strings) into lists.
    
    Args:
        raw_list: List of raw data dictionaries
        
    Returns:
        Dictionary with lists of strings
    """
    if not raw_list:
        return {}
    
    # Get all keys
    all_keys = set()
    for item in raw_list:
        if isinstance(item, dict):
            all_keys.update(item.keys())
    
    collated = {}
    for key in all_keys:
        collated[key] = [item.get(key, '') for item in raw_list if isinstance(item, dict)]
    
    return collated


def collate_generic(value_list: List[Any]) -> Any:
    """
    Generic collation for other data types.
    
    Args:
        value_list: List of values
        
    Returns:
        Collated values (tensor if possible, list otherwise)
    """
    if not value_list:
        return []
    
    # Try to convert to tensor
    try:
        if isinstance(value_list[0], (int, float)):
            return torch.tensor(value_list)
        elif isinstance(value_list[0], torch.Tensor):
            return torch.stack(value_list, dim=0)
    except Exception:
        pass
    
    # Return as list
    return value_list


class AdaptiveCollator:
    """
    Adaptive collator that handles different batch sizes and modalities.
    """
    
    def __init__(self, 
                 max_sequence_length: int = 512,
                 pad_token_id: int = 0,
                 dynamic_padding: bool = True):
        """
        Initialize adaptive collator.
        
        Args:
            max_sequence_length: Maximum sequence length
            pad_token_id: Padding token ID
            dynamic_padding: Whether to use dynamic padding
        """
        self.max_sequence_length = max_sequence_length
        self.pad_token_id = pad_token_id
        self.dynamic_padding = dynamic_padding
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate batch with adaptive padding.
        
        Args:
            batch: List of samples
            
        Returns:
            Collated batch
        """
        if not batch:
            return {}
        
        # Use standard collation first
        collated = multimodal_collate_fn(batch)
        
        # Apply adaptive padding if enabled
        if self.dynamic_padding:
            collated = self._apply_dynamic_padding(collated)
        
        return collated
    
    def _apply_dynamic_padding(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Apply dynamic padding to reduce memory usage."""
        
        # For token sequences, find actual maximum length
        for key in ['text_tokens', 'smiles_tokens', 'scaffold_tokens', 'target_tokens']:
            if key in batch and 'input_ids' in batch[key]:
                input_ids = batch[key]['input_ids']
                
                # Find actual sequence lengths
                if 'attention_mask' in batch[key]:
                    actual_lengths = batch[key]['attention_mask'].sum(dim=1)
                    max_actual_length = int(actual_lengths.max().item())
                else:
                    # Find max length without padding tokens
                    mask = input_ids != self.pad_token_id
                    max_actual_length = mask.sum(dim=1).max().item()
                
                # Trim to actual maximum length
                max_length = min(max_actual_length, self.max_sequence_length)
                batch[key]['input_ids'] = input_ids[:, :max_length]
                
                if 'attention_mask' in batch[key]:
                    batch[key]['attention_mask'] = batch[key]['attention_mask'][:, :max_length]
        
        return batch


class ConversationalCollator:
    """
    Specialized collator for conversational molecular design.
    """
    
    def __init__(self, 
                 max_context_length: int = 1024,
                 max_turns: int = 5):
        """
        Initialize conversational collator.
        
        Args:
            max_context_length: Maximum context sequence length
            max_turns: Maximum conversation turns
        """
        self.max_context_length = max_context_length
        self.max_turns = max_turns
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate conversational batch.
        
        Args:
            batch: List of conversation samples
            
        Returns:
            Collated conversational batch
        """
        # Start with standard collation
        collated = multimodal_collate_fn(batch)
        
        # Add conversation-specific processing
        if 'conversation_history' in batch[0]:
            collated['conversation_history'] = self._collate_conversations(
                [sample['conversation_history'] for sample in batch]
            )
        
        return collated
    
    def _collate_conversations(self, conversations: List[List[Dict]]) -> Dict[str, Any]:
        """Collate conversation histories."""
        # Implementation for conversation history collation
        # This would depend on the specific conversation format
        return {'histories': conversations}


def create_data_loader(dataset, 
                      batch_size: int = 32,
                      shuffle: bool = True,
                      num_workers: int = 0,
                      collate_type: str = 'standard',
                      **kwargs) -> torch.utils.data.DataLoader:
    """
    Create data loader with appropriate collate function.
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        collate_type: Type of collation ('standard', 'adaptive', 'conversational')
        **kwargs: Additional DataLoader arguments
        
    Returns:
        Configured DataLoader
    """
    # Select collate function
    if collate_type == 'adaptive':
        collate_fn = AdaptiveCollator(**kwargs)
    elif collate_type == 'conversational':
        collate_fn = ConversationalCollator(**kwargs)
    else:
        collate_fn = multimodal_collate_fn
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        **{k: v for k, v in kwargs.items() if k not in ['max_sequence_length', 'pad_token_id', 'dynamic_padding', 'max_context_length', 'max_turns']}
    )