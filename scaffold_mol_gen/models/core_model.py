"""
Core model implementation for scaffold-based molecular generation.

This module integrates multi-modal encoders, fusion layers, and MolT5
to create a comprehensive molecular generation system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import Dict, Any, Optional, Union, List, Tuple
import logging

from .encoders import MultiModalEncoder  # 使用新的多模态编码器
from .fusion import AdvancedModalFusion
from .decoders import SMILESDecoder, GraphDecoder, ImageDecoder

logger = logging.getLogger(__name__)

class ScaffoldBasedMolT5Generator(nn.Module):
    """
    Scaffold-based multi-modal molecular generation model.
    
    This model combines multi-modal encoders with MolT5 for generating
    molecules while preserving scaffold structures.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.hidden_size = config.get('hidden_size', 768)
        
        # Multi-modal encoders
        self.encoders = EnhancedMultiModalEncoders(config)
        
        # Modal fusion layer  
        self.fusion_layer = AdvancedModalFusion(config)
        
        # MolT5 core model
        self.molt5_checkpoint = config.get('molt5_checkpoint', 'laituan245/molt5-large')
        self.molt5 = T5ForConditionalGeneration.from_pretrained(self.molt5_checkpoint)
        self.molt5_tokenizer = T5Tokenizer.from_pretrained(self.molt5_checkpoint)
        
        # Adapter layers for different tasks
        self.task_adapters = nn.ModuleDict({
            'smiles': self._create_adapter('smiles'),
            'graph': self._create_adapter('graph'),
            'image': self._create_adapter('image')
        })
        
        # Task-specific decoders
        self.decoders = nn.ModuleDict({
            'smiles': nn.Identity(),  # Use MolT5 directly
            'graph': GraphDecoder(config),
            'image': ImageDecoder(config)
        })
        
        # Generation parameters
        self.generation_config = config.get('generation', {})
        self.max_length = self.generation_config.get('max_length', 200)
        self.num_beams = self.generation_config.get('num_beams', 5)
        self.temperature = self.generation_config.get('temperature', 0.8)
        self.top_k = self.generation_config.get('top_k', 50)
        self.top_p = self.generation_config.get('top_p', 0.95)
        
    def _create_adapter(self, task_type: str) -> nn.Module:
        """Create adapter layer for specific task"""
        return nn.Sequential(
            nn.Linear(self.hidden_size, self.molt5.config.d_model),
            nn.LayerNorm(self.molt5.config.d_model),
            nn.Dropout(self.config.get('dropout', 0.1))
        )
    
    def encode_inputs(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Encode multi-modal inputs using specialized encoders.
        
        Args:
            batch: Dictionary containing input data for different modalities
            
        Returns:
            encoded_features: Dictionary of encoded features per modality
        """
        # Prepare inputs for encoders
        encoder_inputs = {}
        
        # Process scaffold inputs (can be SMILES, graph, or image)
        if 'scaffold_tokens' in batch:
            encoder_inputs['smiles'] = batch['scaffold_tokens']
        elif 'scaffold_graph' in batch:
            encoder_inputs['graph'] = batch['scaffold_graph']
        elif 'scaffold_image' in batch:
            encoder_inputs['image'] = batch['scaffold_image']
        
        # Process text input
        if 'text_tokens' in batch:
            encoder_inputs['text'] = batch['text_tokens']
        
        # Process additional modalities
        if 'additional_graph' in batch:
            encoder_inputs['graph'] = batch['additional_graph']
        if 'additional_image' in batch:
            encoder_inputs['image'] = batch['additional_image']
        
        # Encode all modalities
        encoded_features = self.encoders(encoder_inputs)
        
        return encoded_features
    
    def fuse_modalities(self, encoded_features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Fuse multi-modal features using advanced fusion mechanisms.
        
        Args:
            encoded_features: Dictionary of encoded features per modality
            
        Returns:
            fused_features: Unified representation [batch_size, hidden_size]
            fusion_info: Additional information about fusion process
        """
        fused_features, fusion_info = self.fusion_layer(encoded_features)
        return fused_features, fusion_info
    
    def prepare_molt5_inputs(self, fused_features: torch.Tensor, 
                            attention_mask: Optional[torch.Tensor] = None) -> Any:
        """
        Prepare inputs for MolT5 model.
        
        Args:
            fused_features: Fused multi-modal features [batch_size, hidden_size]
            attention_mask: Optional attention mask
            
        Returns:
            encoder_outputs: Formatted for MolT5 consumption
        """
        from transformers.modeling_outputs import BaseModelOutput
        
        batch_size = fused_features.shape[0]
        seq_length = 1  # Single aggregated representation
        
        # Expand to sequence format for T5
        hidden_states = fused_features.unsqueeze(1)  # [batch, 1, hidden]
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, device=fused_features.device)
        
        # Create proper encoder outputs using BaseModelOutput
        encoder_outputs = BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=None,
            attentions=None
        )
        
        return encoder_outputs
    
    def forward(self, batch: Dict[str, Any], 
                output_modality: str = 'smiles',
                mode: str = 'train') -> Union[torch.Tensor, List[str]]:
        """
        Forward pass of the model.
        
        Args:
            batch: Input batch containing multi-modal data
            output_modality: Target output modality ('smiles', 'graph', 'image')
            mode: 'train' or 'inference'
            
        Returns:
            outputs: Model outputs (logits for training, generated sequences for inference)
        """
        # 1. Encode multi-modal inputs
        encoded_features = self.encode_inputs(batch)
        
        if not encoded_features:
            raise ValueError("No valid input features found in batch")
        
        # 2. Fuse modalities
        fused_features, fusion_info = self.fuse_modalities(encoded_features)
        
        # 3. Task-specific processing
        if output_modality == 'smiles':
            return self._generate_smiles(fused_features, batch, mode)
        elif output_modality == 'graph':
            return self._generate_graph(fused_features, batch, mode)
        elif output_modality == 'image':
            return self._generate_image(fused_features, batch, mode)
        else:
            raise ValueError(f"Unsupported output modality: {output_modality}")
    
    def _generate_smiles(self, fused_features: torch.Tensor, 
                        batch: Dict[str, Any], mode: str) -> Union[torch.Tensor, List[str]]:
        """Generate SMILES using MolT5"""
        # Adapt features for MolT5
        adapted_features = self.task_adapters['smiles'](fused_features)
        
        # Prepare encoder outputs
        encoder_outputs = self.prepare_molt5_inputs(adapted_features)
        
        if mode == 'train' and 'target_tokens' in batch:
            # Training mode: compute loss
            target_input_ids = batch['target_tokens']['input_ids']
            target_attention_mask = batch['target_tokens']['attention_mask']
            
            # Shift targets for causal language modeling
            decoder_input_ids = target_input_ids[:, :-1].contiguous()
            labels = target_input_ids[:, 1:].contiguous()
            
            outputs = self.molt5(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                labels=labels
            )
            
            # Return the loss directly from T5 model
            return outputs.loss
            
        else:
            # Inference mode: generate
            generated_ids = self.molt5.generate(
                encoder_outputs=encoder_outputs,
                max_length=self.max_length,
                num_beams=self.num_beams,
                temperature=self.temperature,
                do_sample=True,
                top_k=self.top_k,
                top_p=self.top_p,
                pad_token_id=self.molt5_tokenizer.pad_token_id,
                eos_token_id=self.molt5_tokenizer.eos_token_id
            )
            
            # Decode to SMILES strings
            generated_smiles = []
            for ids in generated_ids:
                smiles = self.molt5_tokenizer.decode(ids, skip_special_tokens=True)
                generated_smiles.append(smiles)
            
            return generated_smiles
    
    def _generate_graph(self, fused_features: torch.Tensor, 
                       batch: Dict[str, Any], mode: str) -> torch.Tensor:
        """Generate molecular graph using graph decoder"""
        adapted_features = self.task_adapters['graph'](fused_features)
        return self.decoders['graph'](adapted_features, batch, mode)
    
    def _generate_image(self, fused_features: torch.Tensor, 
                       batch: Dict[str, Any], mode: str) -> torch.Tensor:
        """Generate molecular image using image decoder"""
        adapted_features = self.task_adapters['image'](fused_features)
        return self.decoders['image'](adapted_features, batch, mode)
    
    def generate(self, scaffold: str, text: str, 
                scaffold_modality: str = 'smiles',
                output_modality: str = 'smiles',
                num_samples: int = 1,
                **generation_kwargs) -> List[str]:
        """
        High-level generation interface.
        
        Args:
            scaffold: Scaffold input (SMILES string, image path, etc.)
            text: Text description
            scaffold_modality: Input modality for scaffold
            output_modality: Desired output modality
            num_samples: Number of samples to generate
            **generation_kwargs: Additional generation parameters
            
        Returns:
            generated_samples: List of generated molecules/representations
        """
        self.eval()
        
        # Prepare batch
        batch = self._prepare_generation_batch(
            scaffold, text, scaffold_modality
        )
        
        # Update generation parameters
        for key, value in generation_kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        generated_samples = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                outputs = self.forward(batch, output_modality, mode='inference')
                if isinstance(outputs, list):
                    generated_samples.extend(outputs)
                else:
                    generated_samples.append(outputs)
        
        return generated_samples[:num_samples]
    
    def _prepare_generation_batch(self, scaffold: str, text: str, 
                                 scaffold_modality: str) -> Dict[str, Any]:
        """Prepare batch for generation"""
        batch = {}
        device = next(self.parameters()).device
        
        # Process text
        text_encoding = self.molt5_tokenizer(
            text,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(device)
        batch['text_tokens'] = text_encoding
        
        # Process scaffold based on modality
        if scaffold_modality == 'smiles':
            scaffold_encoding = self.molt5_tokenizer(
                scaffold,
                max_length=128,
                padding='max_length', 
                truncation=True,
                return_tensors='pt'
            ).to(device)
            batch['scaffold_tokens'] = scaffold_encoding
            
        elif scaffold_modality == 'graph':
            # Convert SMILES to graph representation
            from ..utils.mol_utils import smiles_to_graph
            graph_data = smiles_to_graph(scaffold)
            if graph_data:
                batch['scaffold_graph'] = graph_data.to(device)
                
        elif scaffold_modality == 'image':
            # Convert SMILES to image representation  
            from ..utils.mol_utils import smiles_to_image
            image_tensor = smiles_to_image(scaffold)
            if image_tensor is not None:
                batch['scaffold_image'] = image_tensor.unsqueeze(0).to(device)
        
        return batch
    
    @classmethod
    def from_pretrained(cls, checkpoint_path: str, config: Optional[Dict] = None):
        """Load model from checkpoint"""
        if config is None:
            # Load config from checkpoint directory
            import os
            config_path = os.path.join(os.path.dirname(checkpoint_path), 'config.yaml')
            if os.path.exists(config_path):
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                # Use default config
                config = cls.get_default_config()
        
        model = cls(config)
        
        # Load state dict
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
        
        return model
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default model configuration"""
        return {
            'hidden_size': 768,
            'num_attention_heads': 12,
            'num_fusion_layers': 6,
            'dropout': 0.1,
            'num_gin_layers': 3,
            'molt5_checkpoint': 'laituan245/molt5-large',
            'generation': {
                'max_length': 200,
                'num_beams': 5,
                'temperature': 0.8,
                'top_k': 50,
                'top_p': 0.95
            }
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'ScaffoldBasedMolT5Generator',
            'version': '1.0.0',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'hidden_size': self.hidden_size,
            'molt5_checkpoint': self.molt5_checkpoint,
            'supported_modalities': ['smiles', 'graph', 'image'],
            'config': self.config
        }