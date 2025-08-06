"""
文本编码器模块
使用BERT或SciBERT处理分子描述文本
"""

import torch
import torch.nn as nn
from transformers import (
    BertModel,
    BertTokenizer,
    AutoModel,
    AutoTokenizer
)
import logging

logger = logging.getLogger(__name__)

class TextEncoder(nn.Module):
    """
    文本编码器，用于编码分子的自然语言描述
    """
    
    def __init__(self,
                 model_name: str = "bert",
                 model_path: str = None,
                 hidden_size: int = 768,
                 max_length: int = 256,
                 freeze_backbone: bool = False,
                 use_pooled: bool = True):
        """
        初始化文本编码器
        
        Args:
            model_name: 模型名称 ("bert" or "scibert")
            model_path: 预训练模型路径
            hidden_size: 输出隐藏层维度
            max_length: 最大序列长度
            freeze_backbone: 是否冻结预训练权重
            use_pooled: 是否使用池化输出
        """
        super().__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.use_pooled = use_pooled
        
        # 加载预训练模型
        self._load_pretrained_model(model_name, model_path)
        
        # 投影层
        encoder_hidden_size = self.encoder.config.hidden_size
        if encoder_hidden_size != hidden_size:
            self.projection = nn.Linear(encoder_hidden_size, hidden_size)
        else:
            self.projection = nn.Identity()
        
        # 额外的文本处理层（可选）
        self.text_enhancement = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 冻结backbone
        if freeze_backbone:
            self._freeze_backbone()
    
    def _load_pretrained_model(self, model_name: str, model_path: str = None):
        """加载预训练模型"""
        if model_name == "bert":
            if model_path is None:
                model_path = "/root/autodl-tmp/text2Mol-models/bert-base-uncased"
            logger.info(f"加载BERT模型: {model_path}")
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.encoder = BertModel.from_pretrained(model_path)
            
        elif model_name == "scibert":
            if model_path is None:
                model_path = "/root/autodl-tmp/text2Mol-models/scibert_scivocab_uncased"
            logger.info(f"加载SciBERT模型: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.encoder = AutoModel.from_pretrained(model_path)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        # 添加化学领域的特殊token
        special_tokens = {
            "additional_special_tokens": [
                "[PROPERTY]", "[/PROPERTY]",
                "[FUNCTION]", "[/FUNCTION]",
                "[TARGET]", "[/TARGET]",
                "[ACTIVITY]", "[/ACTIVITY]",
                "[SOLUBILITY]", "[TOXICITY]", "[BIOAVAILABILITY]"
            ]
        }
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        if num_added > 0:
            self.encoder.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"添加了 {num_added} 个特殊token")
    
    def _freeze_backbone(self):
        """冻结预训练模型参数"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("文本编码器backbone已冻结")
    
    def tokenize(self, texts: list, add_prefix: bool = True):
        """
        对文本进行tokenization
        
        Args:
            texts: 文本列表
            add_prefix: 是否添加任务前缀
            
        Returns:
            tokenized inputs
        """
        # 可选择性地添加任务前缀
        if add_prefix:
            texts = [f"Generate molecule with properties: {text}" for text in texts]
        
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        前向传播
        
        Args:
            input_ids: token ids
            attention_mask: 注意力掩码
            token_type_ids: token类型ids（BERT需要）
            
        Returns:
            encoded_features: 编码后的特征
        """
        # 通过BERT/SciBERT编码
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        if self.use_pooled and hasattr(outputs, 'pooler_output'):
            # 使用[CLS] token的池化输出
            features = outputs.pooler_output
        else:
            # 使用所有token的输出
            features = outputs.last_hidden_state
        
        # 投影到目标维度
        features = self.projection(features)
        
        # 文本增强处理
        if self.use_pooled:
            features = self.text_enhancement(features)
        else:
            # 对序列的每个位置应用增强
            batch_size, seq_len, hidden_size = features.shape
            features = features.reshape(-1, hidden_size)
            features = self.text_enhancement(features)
            features = features.reshape(batch_size, seq_len, hidden_size)
        
        return features
    
    def encode(self, texts: list):
        """
        端到端编码接口
        
        Args:
            texts: 文本列表
            
        Returns:
            encoded_features: 编码后的特征
        """
        # Tokenize
        inputs = self.tokenize(texts)
        
        # Move to device
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Encode
        with torch.no_grad():
            encoded_features = self.forward(
                inputs['input_ids'],
                inputs.get('attention_mask'),
                inputs.get('token_type_ids')
            )
        
        return encoded_features
    
    def get_pooled_features(self, encoded_features, attention_mask=None):
        """
        获取池化后的特征表示
        
        Args:
            encoded_features: 编码器输出
            attention_mask: 注意力掩码
            
        Returns:
            pooled_features: 池化后的特征
        """
        if len(encoded_features.shape) == 2:
            # 已经是池化后的特征
            return encoded_features
        
        if attention_mask is None:
            # 简单平均池化
            return encoded_features.mean(dim=1)
        else:
            # 根据mask进行平均池化
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = (encoded_features * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            return sum_embeddings / sum_mask


class BERTEncoder(TextEncoder):
    """标准BERT编码器"""
    def __init__(self, **kwargs):
        kwargs['model_name'] = 'bert'
        super().__init__(**kwargs)


class SciBERTEncoder(TextEncoder):
    """SciBERT编码器（科学文本）"""
    def __init__(self, **kwargs):
        kwargs['model_name'] = 'scibert'
        super().__init__(**kwargs)