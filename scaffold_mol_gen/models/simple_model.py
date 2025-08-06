"""
Simplified end-to-end molecular generation model for training.
"""

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config, BertModel
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

class End2EndMolecularGenerator(nn.Module):
    """
    Simplified molecular generation model combining:
    - Text encoder (BERT)
    - Scaffold encoder (shared MolT5 encoder)
    - Fusion layer
    - MolT5 decoder for generation
    """
    
    def __init__(self,
                 hidden_size: int = 768,
                 num_attention_heads: int = 12,
                 num_fusion_layers: int = 3,
                 dropout: float = 0.1,
                 use_gradient_checkpointing: bool = False,
                 device: str = 'cuda',
                 molt5_path: str = "/root/autodl-tmp/text2Mol-models/MolT5-Large-Caption2SMILES",
                 bert_path: str = "/root/autodl-tmp/text2Mol-models/bert-base-uncased"):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.device = device
        
        logger.info("Initializing End2EndMolecularGenerator...")
        
        # Load pretrained models
        logger.info("Loading BERT text encoder...")
        self.text_encoder = BertModel.from_pretrained(bert_path)
        
        logger.info("Loading MolT5 model...")
        self.molt5 = T5ForConditionalGeneration.from_pretrained(molt5_path)
        
        # Get MolT5 dimensions
        molt5_dim = self.molt5.config.d_model  # Usually 1024 for MolT5-Large
        
        # Projection layers to align dimensions
        self.text_projection = nn.Linear(768, hidden_size)  # BERT -> hidden
        self.scaffold_projection = nn.Linear(molt5_dim, hidden_size)  # MolT5 -> hidden
        
        # Fusion layers
        self.fusion_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_attention_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_fusion_layers)
        ])
        
        # Final projection to MolT5 dimension
        self.output_projection = nn.Linear(hidden_size, molt5_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Gradient checkpointing - Always enable to save memory
        if True:  # Always enable for memory efficiency
            if hasattr(self.text_encoder, 'gradient_checkpointing_enable'):
                self.text_encoder.gradient_checkpointing_enable()
            if hasattr(self.molt5, 'gradient_checkpointing_enable'):
                self.molt5.gradient_checkpointing_enable()
        
        # Freeze some layers to save memory and speed up training
        self._freeze_layers()
        
    def _freeze_layers(self):
        """Freeze pretrained layers based on configuration."""
        # Freeze most of BERT except last 2 layers
        for param in self.text_encoder.embeddings.parameters():
            param.requires_grad = False
        for layer in self.text_encoder.encoder.layer[:-2]:
            for param in layer.parameters():
                param.requires_grad = False
        
        # Freeze MolT5 embeddings
        for param in self.molt5.shared.parameters():
            param.requires_grad = False
        
        # Freeze early MolT5 encoder layers
        if hasattr(self.molt5.encoder, 'block'):
            for layer in self.molt5.encoder.block[:6]:  # Freeze first 6 layers
                for param in layer.parameters():
                    param.requires_grad = False
    
    def encode_text(self, text_input_ids, text_attention_mask):
        """Encode text using BERT."""
        outputs = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask
        )
        # Use pooled output or mean of sequence
        text_features = outputs.last_hidden_state.mean(dim=1)  # [batch, 768]
        text_features = self.text_projection(text_features)  # [batch, hidden]
        return text_features
    
    def encode_scaffold(self, scaffold_input_ids, scaffold_attention_mask):
        """Encode scaffold SMILES using MolT5 encoder."""
        # Get MolT5 encoder outputs
        encoder_outputs = self.molt5.encoder(
            input_ids=scaffold_input_ids,
            attention_mask=scaffold_attention_mask
        )
        # Pool encoder outputs
        scaffold_features = encoder_outputs.last_hidden_state.mean(dim=1)  # [batch, molt5_dim]
        scaffold_features = self.scaffold_projection(scaffold_features)  # [batch, hidden]
        return scaffold_features
    
    def fuse_features(self, text_features, scaffold_features):
        """Fuse text and scaffold features."""
        # Combine features
        combined = torch.stack([text_features, scaffold_features], dim=1)  # [batch, 2, hidden]
        
        # Apply fusion layers
        for layer in self.fusion_layers:
            combined = layer(combined)
        
        # Pool to single vector
        fused = combined.mean(dim=1)  # [batch, hidden]
        return fused
    
    def forward(self, 
                text_input_ids=None,
                text_attention_mask=None,
                scaffold_input_ids=None,
                scaffold_attention_mask=None,
                labels=None,
                **kwargs):
        """
        Forward pass for training.
        
        Returns:
            dict with 'loss' and optionally 'logits'
        """
        # Encode text
        if text_input_ids is not None:
            text_features = self.encode_text(text_input_ids, text_attention_mask)
        else:
            text_features = torch.zeros(scaffold_input_ids.size(0), self.hidden_size).to(self.device)
        
        # Encode scaffold
        if scaffold_input_ids is not None:
            scaffold_features = self.encode_scaffold(scaffold_input_ids, scaffold_attention_mask)
        else:
            scaffold_features = torch.zeros(text_input_ids.size(0), self.hidden_size).to(self.device)
        
        # Fuse features
        fused_features = self.fuse_features(text_features, scaffold_features)
        
        # Project to MolT5 dimension
        molt5_features = self.output_projection(fused_features)  # [batch, molt5_dim]
        
        # Expand to sequence for decoder
        batch_size = molt5_features.size(0)
        # Create a simple encoder output by repeating the features
        encoder_hidden_states = molt5_features.unsqueeze(1).repeat(1, 10, 1)  # [batch, 10, molt5_dim]
        
        # Prepare decoder inputs
        if labels is not None:
            # Shift labels for decoder input
            decoder_input_ids = labels.clone()
            decoder_input_ids[:, 1:] = labels[:, :-1]
            decoder_input_ids[:, 0] = self.molt5.config.decoder_start_token_id
            
            # Forward through MolT5 decoder with proper encoder outputs
            from transformers.modeling_outputs import BaseModelOutput
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
            
            outputs = self.molt5(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                return_dict=True
            )
            
            loss = outputs.loss
            logits = outputs.logits
        else:
            loss = None
            logits = None
        
        return {
            'loss': loss,
            'logits': logits
        }
    
    def generate(self, 
                 text_input_ids=None,
                 text_attention_mask=None,
                 scaffold_input_ids=None,
                 scaffold_attention_mask=None,
                 max_length=128,
                 num_beams=5,
                 temperature=0.8,
                 **kwargs):
        """
        Generate molecules given text and scaffold.
        """
        # Encode inputs
        if text_input_ids is not None:
            text_features = self.encode_text(text_input_ids, text_attention_mask)
        else:
            text_features = torch.zeros(1, self.hidden_size).to(self.device)
        
        if scaffold_input_ids is not None:
            scaffold_features = self.encode_scaffold(scaffold_input_ids, scaffold_attention_mask)
        else:
            scaffold_features = torch.zeros(1, self.hidden_size).to(self.device)
        
        # Fuse and project
        fused_features = self.fuse_features(text_features, scaffold_features)
        molt5_features = self.output_projection(fused_features)
        
        # Create encoder outputs
        encoder_hidden_states = molt5_features.unsqueeze(1).repeat(1, 10, 1)
        
        # Generate using MolT5 with proper encoder outputs
        from transformers.modeling_outputs import BaseModelOutput
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
        
        outputs = self.molt5.generate(
            encoder_outputs=encoder_outputs,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            do_sample=True,
            **kwargs
        )
        
        return outputs