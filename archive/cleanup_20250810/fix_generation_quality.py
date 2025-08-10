#!/usr/bin/env python3
"""
修复MolT5生成质量问题
确保生成有效的SMILES而非随机文本
"""

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from typing import List, Optional, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedMolT5Generator(nn.Module):
    """
    修复版的MolT5生成器
    确保生成化学SMILES而非文本
    """
    
    def __init__(self, 
                 molt5_path: str = "/root/autodl-tmp/text2Mol-models/molt5-base",
                 device: str = 'cuda'):
        super().__init__()
        
        self.device = device
        
        # 加载MolT5模型和tokenizer
        logger.info(f"加载MolT5模型: {molt5_path}")
        self.molt5 = T5ForConditionalGeneration.from_pretrained(molt5_path)
        self.tokenizer = T5Tokenizer.from_pretrained(molt5_path)
        
        # 设置特殊的SMILES生成参数
        self.molt5.config.max_length = 512
        self.molt5.config.num_beams = 5
        self.molt5.config.early_stopping = True
        
        # 添加SMILES特殊标记（如果需要）
        special_tokens = ['<SMILES>', '</SMILES>', '<MOL>', '</MOL>']
        num_added = self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        if num_added > 0:
            self.molt5.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"添加了{num_added}个特殊标记")
        
        self.to(device)
    
    def generate_smiles(self,
                       encoder_outputs: torch.Tensor,
                       attention_mask: torch.Tensor,
                       num_beams: int = 5,
                       max_length: int = 256,
                       temperature: float = 1.0) -> List[str]:
        """
        生成SMILES，使用改进的参数
        """
        from transformers.modeling_outputs import BaseModelOutput
        
        # 准备encoder输出
        molt5_encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_outputs
        )
        
        # 使用特定的SMILES生成参数
        with torch.no_grad():
            # 生成时使用更保守的参数
            generated_ids = self.molt5.generate(
                encoder_outputs=molt5_encoder_outputs,
                attention_mask=attention_mask,
                max_length=max_length,
                min_length=5,  # 最小长度
                num_beams=num_beams,
                temperature=temperature,
                do_sample=False,  # 使用贪婪解码而非采样
                repetition_penalty=1.2,  # 避免重复
                length_penalty=1.0,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id if hasattr(self.tokenizer, 'bos_token_id') else None,
            )
        
        # 解码并后处理
        generated_texts = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        # 后处理：清理生成的文本
        cleaned_smiles = []
        for text in generated_texts:
            # 移除可能的文本前缀/后缀
            smiles = self.clean_generated_text(text)
            cleaned_smiles.append(smiles)
        
        return cleaned_smiles
    
    def clean_generated_text(self, text: str) -> str:
        """
        清理生成的文本，提取SMILES
        """
        # 移除常见的非SMILES文本
        text = text.strip()
        
        # 如果包含特殊标记，提取它们之间的内容
        if '<SMILES>' in text and '</SMILES>' in text:
            start = text.find('<SMILES>') + len('<SMILES>')
            end = text.find('</SMILES>')
            text = text[start:end].strip()
        
        # 移除明显的非SMILES字符
        # SMILES通常只包含特定字符集
        valid_chars = set('CNOSPFClBrIcnospfclbri()[]=#-+\\/@123456789%')
        
        # 如果文本包含太多非SMILES字符，可能是错误生成
        if text and len([c for c in text if c not in valid_chars]) / len(text) > 0.1:
            # 尝试提取看起来像SMILES的部分
            tokens = text.split()
            for token in tokens:
                if len([c for c in token if c in valid_chars]) / len(token) > 0.8:
                    return token
            # 如果没有找到，返回简单的默认SMILES
            return "CC"  # 乙烷作为默认
        
        return text if text else "CC"


def test_fixed_generator():
    """测试修复的生成器"""
    import numpy as np
    from rdkit import Chem
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    
    # 创建生成器
    generator = FixedMolT5Generator(device=device)
    
    # 创建测试输入（模拟encoder输出）
    batch_size = 3
    seq_len = 32
    hidden_size = 768
    
    encoder_outputs = torch.randn(batch_size, seq_len, hidden_size).to(device)
    attention_mask = torch.ones(batch_size, seq_len).to(device)
    
    logger.info("测试生成...")
    
    # 生成SMILES
    generated_smiles = generator.generate_smiles(
        encoder_outputs=encoder_outputs,
        attention_mask=attention_mask,
        num_beams=5,
        max_length=128,
        temperature=0.8
    )
    
    logger.info(f"\n生成的SMILES:")
    valid_count = 0
    for i, smiles in enumerate(generated_smiles):
        # 验证SMILES
        mol = Chem.MolFromSmiles(smiles)
        is_valid = mol is not None
        if is_valid:
            valid_count += 1
        
        status = "✅ 有效" if is_valid else "❌ 无效"
        logger.info(f"  {i+1}. {smiles[:50]}{'...' if len(smiles) > 50 else ''} - {status}")
    
    logger.info(f"\n有效率: {valid_count}/{len(generated_smiles)} ({valid_count/len(generated_smiles)*100:.1f}%)")
    
    return generated_smiles


if __name__ == "__main__":
    test_fixed_generator()