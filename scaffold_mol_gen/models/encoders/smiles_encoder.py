"""
SMILES编码器模块
使用MolT5或BERT处理Scaffold SMILES输入
"""

import torch
import torch.nn as nn
from transformers import (
    T5EncoderModel,
    T5Tokenizer,
    AutoModel,
    AutoTokenizer
)
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class SMILESEncoder(nn.Module):
    """
    SMILES编码器，用于编码Scaffold SMILES
    支持MolT5和BERT两种backbone
    """
    
    def __init__(self, 
                 model_type: str = "molt5",
                 model_path: str = None,
                 hidden_size: int = 768,
                 max_length: int = 128,
                 freeze_backbone: bool = False):
        """
        初始化SMILES编码器
        
        Args:
            model_type: 模型类型 ("molt5" or "bert")
            model_path: 预训练模型路径
            hidden_size: 输出隐藏层维度
            max_length: 最大序列长度
            freeze_backbone: 是否冻结预训练权重
        """
        super().__init__()
        
        self.model_type = model_type
        self.max_length = max_length
        self.hidden_size = hidden_size
        
        # 加载预训练模型
        if model_type == "molt5":
            self._init_molt5(model_path)
        elif model_type == "bert":
            self._init_bert(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # 投影层，将编码器输出映射到统一维度
        encoder_hidden_size = self.encoder.config.hidden_size
        if encoder_hidden_size != hidden_size:
            self.projection = nn.Linear(encoder_hidden_size, hidden_size)
        else:
            self.projection = nn.Identity()
        
        # 是否冻结backbone
        if freeze_backbone:
            self._freeze_backbone()
    
    def _init_molt5(self, model_path: str = None):
        """初始化MolT5编码器"""
        if model_path is None:
            model_path = "/root/autodl-tmp/text2Mol-models/MolT5-Large-Caption2SMILES"
        
        logger.info(f"加载MolT5编码器: {model_path}")
        
        # 加载tokenizer和encoder
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        # 只加载encoder部分
        self.encoder = T5EncoderModel.from_pretrained(model_path)
        
        # 添加特殊token用于Scaffold标记
        special_tokens = {
            "additional_special_tokens": [
                "[SCAFFOLD]", "[/SCAFFOLD]",
                "[FUNCTIONAL_GROUP]", "[/FUNCTIONAL_GROUP]"
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.encoder.resize_token_embeddings(len(self.tokenizer))
    
    def _init_bert(self, model_path: str = None):
        """初始化BERT编码器"""
        if model_path is None:
            model_path = "/root/autodl-tmp/text2Mol-models/bert-base-uncased"
        
        logger.info(f"加载BERT编码器: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.encoder = AutoModel.from_pretrained(model_path)
        
        # 添加化学相关的特殊token
        special_tokens = {
            "additional_special_tokens": [
                "[SCAFFOLD]", "[/SCAFFOLD]",
                "[SMILES]", "[/SMILES]",
                "[RING]", "[CHAIN]", "[BRANCH]"
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.encoder.resize_token_embeddings(len(self.tokenizer))
    
    def _freeze_backbone(self):
        """冻结预训练模型参数"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("预训练模型参数已冻结")
    
    def tokenize(self, smiles_list: list, add_special_tokens: bool = True):
        """
        对SMILES进行tokenization
        
        Args:
            smiles_list: SMILES字符串列表
            add_special_tokens: 是否添加特殊token
            
        Returns:
            tokenized inputs
        """
        # 为Scaffold SMILES添加特殊标记
        if add_special_tokens:
            smiles_list = [f"[SCAFFOLD] {s} [/SCAFFOLD]" for s in smiles_list]
        
        return self.tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
    
    def forward(self, smiles_input, attention_mask=None):
        """
        前向传播
        
        Args:
            smiles_input: tokenized SMILES input ids
            attention_mask: 注意力掩码
            
        Returns:
            encoded_features: [batch_size, seq_len, hidden_size]
        """
        # 编码
        if self.model_type == "molt5":
            encoder_outputs = self.encoder(
                input_ids=smiles_input,
                attention_mask=attention_mask
            )
            encoded_features = encoder_outputs.last_hidden_state
        else:  # BERT
            encoder_outputs = self.encoder(
                input_ids=smiles_input,
                attention_mask=attention_mask
            )
            encoded_features = encoder_outputs.last_hidden_state
        
        # 投影到目标维度
        encoded_features = self.projection(encoded_features)
        
        return encoded_features
    
    def encode(self, smiles_list: list):
        """
        端到端编码接口
        
        Args:
            smiles_list: SMILES字符串列表
            
        Returns:
            encoded_features: 编码后的特征
        """
        # Tokenize
        inputs = self.tokenize(smiles_list)
        
        # Move to device
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Encode
        with torch.no_grad():
            encoded_features = self.forward(
                inputs['input_ids'],
                inputs.get('attention_mask')
            )
        
        return encoded_features
    
    def get_pooled_features(self, encoded_features, attention_mask=None):
        """
        获取池化后的特征表示
        
        Args:
            encoded_features: 编码器输出 [batch, seq_len, hidden]
            attention_mask: 注意力掩码
            
        Returns:
            pooled_features: [batch, hidden]
        """
        if attention_mask is None:
            # 简单平均池化
            return encoded_features.mean(dim=1)
        else:
            # 根据mask进行平均池化
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = (encoded_features * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            return sum_embeddings / sum_mask


class BartSMILESEncoder(SMILESEncoder):
    """
    BartSMILES编码器（使用MolT5作为替代）
    注：由于BartSMILES模型难以获取，这里使用MolT5作为实现
    """
    
    def __init__(self, **kwargs):
        kwargs['model_type'] = 'molt5'
        super().__init__(**kwargs)
        logger.info("使用MolT5作为BartSMILES的替代实现")