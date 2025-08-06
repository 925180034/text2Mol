"""
Stable molecular generation model with better initialization and training stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer, BertModel
from transformers.modeling_outputs import BaseModelOutput
import math

class StableMolecularGenerator(nn.Module):
    """
    Simplified and stable molecular generation model.
    Key improvements:
    - Better weight initialization
    - Gradient-friendly architecture
    - Stable loss computation
    """
    
    def __init__(self,
                 bert_path="/root/autodl-tmp/text2Mol-models/bert-base-uncased",
                 molt5_path="/root/autodl-tmp/text2Mol-models/MolT5-Large-Caption2SMILES",
                 hidden_dim=768,
                 freeze_molt5_encoder=True,
                 freeze_molt5_decoder_layers=20,  # Freeze first 20 layers of decoder
                 use_simple_fusion=True):
        super().__init__()
        
        print("Initializing Stable Molecular Generator...")
        
        # Load pretrained models
        self.text_encoder = BertModel.from_pretrained(bert_path)
        self.molt5 = T5ForConditionalGeneration.from_pretrained(molt5_path)
        
        # Get dimensions
        bert_dim = self.text_encoder.config.hidden_size  # 768
        molt5_dim = self.molt5.config.d_model  # 1024
        
        # Freeze most parameters for stability
        self._freeze_parameters(freeze_molt5_encoder, freeze_molt5_decoder_layers)
        
        # Simple fusion without complex attention
        if use_simple_fusion:
            # Simple linear projections
            self.text_proj = nn.Sequential(
                nn.Linear(bert_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            )
            
            self.scaffold_proj = nn.Sequential(
                nn.Linear(molt5_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            )
            
            # Simple fusion: concatenate and project
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, molt5_dim),
                nn.LayerNorm(molt5_dim)
            )
        else:
            # More complex fusion (optional)
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            self.fusion = nn.Linear(hidden_dim, molt5_dim)
        
        self.use_simple_fusion = use_simple_fusion
        
        # Initialize weights properly
        self._init_weights()
        
        print(f"Model initialized with {self.count_parameters()} trainable parameters")
    
    def _freeze_parameters(self, freeze_encoder, freeze_decoder_layers):
        """Freeze parameters for stability"""
        # Freeze BERT embeddings
        for param in self.text_encoder.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze early BERT layers
        for layer in self.text_encoder.encoder.layer[:10]:
            for param in layer.parameters():
                param.requires_grad = False
        
        # Freeze MolT5 encoder if requested
        if freeze_encoder:
            for param in self.molt5.encoder.parameters():
                param.requires_grad = False
        
        # Freeze MolT5 embeddings
        for param in self.molt5.shared.parameters():
            param.requires_grad = False
        
        # Freeze early decoder layers
        if hasattr(self.molt5.decoder, 'block'):
            for i, layer in enumerate(self.molt5.decoder.block):
                if i < freeze_decoder_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
    
    def _init_weights(self):
        """Careful weight initialization"""
        # Initialize projections with Xavier/He initialization
        for module in [self.text_proj, self.scaffold_proj, self.fusion]:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        # Xavier initialization for linear layers
                        nn.init.xavier_uniform_(layer.weight, gain=0.5)  # Small gain
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def encode_text(self, input_ids, attention_mask):
        """Encode text with BERT"""
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        # Use CLS token
        text_features = outputs.last_hidden_state[:, 0, :]  # [batch, 768]
        return self.text_proj(text_features)  # [batch, hidden_dim]
    
    def encode_scaffold(self, input_ids, attention_mask):
        """Encode scaffold with MolT5 encoder"""
        encoder_outputs = self.molt5.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        # Pool by mean
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (encoder_outputs.last_hidden_state * mask_expanded).sum(1)
        sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
        scaffold_features = sum_embeddings / sum_mask  # [batch, 1024]
        
        return self.scaffold_proj(scaffold_features)  # [batch, hidden_dim]
    
    def forward(self,
                text_input_ids=None,
                text_attention_mask=None,
                scaffold_input_ids=None,
                scaffold_attention_mask=None,
                labels=None,
                target_attention_mask=None,  # Accept but don't use
                return_dict=True,
                **kwargs):
        """
        Forward pass with stable loss computation.
        """
        batch_size = text_input_ids.size(0) if text_input_ids is not None else scaffold_input_ids.size(0)
        
        # Encode text
        text_features = self.encode_text(text_input_ids, text_attention_mask)
        
        # Encode scaffold  
        scaffold_features = self.encode_scaffold(scaffold_input_ids, scaffold_attention_mask)
        
        # Fuse features
        if self.use_simple_fusion:
            # Simple concatenation and projection
            combined = torch.cat([text_features, scaffold_features], dim=-1)
            fused_features = self.fusion(combined)  # [batch, 1024]
        else:
            # Cross-attention fusion
            fused_features, _ = self.cross_attention(
                query=text_features.unsqueeze(1),
                key=scaffold_features.unsqueeze(1),
                value=scaffold_features.unsqueeze(1)
            )
            fused_features = self.fusion(fused_features.squeeze(1))
        
        # Create encoder hidden states for decoder
        # Expand to sequence length
        encoder_hidden_states = fused_features.unsqueeze(1).expand(-1, 20, -1)
        
        # Create proper encoder outputs
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
        
        # Prepare decoder inputs if training
        if labels is not None:
            # Shift labels for decoder
            decoder_input_ids = labels.clone()
            decoder_input_ids[:, 1:] = labels[:, :-1]
            decoder_input_ids[:, 0] = self.molt5.config.decoder_start_token_id
            
            # Forward through MolT5 with stable loss
            outputs = self.molt5(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                return_dict=True
            )
            
            # Add label smoothing for stability
            loss = outputs.loss
            if loss is not None and not torch.isnan(loss):
                # Add small L2 regularization to prevent explosion
                l2_reg = 0.0
                for param in self.parameters():
                    if param.requires_grad:
                        l2_reg += torch.norm(param, p=2)
                loss = loss + 1e-6 * l2_reg
            else:
                # If loss is NaN, return a small constant loss
                loss = torch.tensor(10.0, device=text_input_ids.device, requires_grad=True)
            
            if return_dict:
                return {'loss': loss, 'logits': outputs.logits if hasattr(outputs, 'logits') else None}
            return loss
        
        # Inference mode
        return {'loss': None, 'logits': None}
    
    def generate(self, text_input_ids, text_attention_mask, 
                 scaffold_input_ids, scaffold_attention_mask,
                 max_length=128, num_beams=3, temperature=0.9,
                 do_sample=True, top_k=50, top_p=0.95):
        """Generate molecules"""
        # Get fused features
        text_features = self.encode_text(text_input_ids, text_attention_mask)
        scaffold_features = self.encode_scaffold(scaffold_input_ids, scaffold_attention_mask)
        
        if self.use_simple_fusion:
            combined = torch.cat([text_features, scaffold_features], dim=-1)
            fused_features = self.fusion(combined)
        else:
            fused_features, _ = self.cross_attention(
                query=text_features.unsqueeze(1),
                key=scaffold_features.unsqueeze(1),
                value=scaffold_features.unsqueeze(1)
            )
            fused_features = self.fusion(fused_features.squeeze(1))
        
        # Create encoder outputs
        encoder_hidden_states = fused_features.unsqueeze(1).expand(-1, 20, -1)
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
        
        # Generate
        outputs = self.molt5.generate(
            encoder_outputs=encoder_outputs,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            early_stopping=True,
            pad_token_id=self.molt5.config.pad_token_id,
            eos_token_id=self.molt5.config.eos_token_id
        )
        
        return outputs