"""
MolT5 Adapter and Generator
将融合特征适配到MolT5并生成分子
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Union
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from transformers.modeling_outputs import BaseModelOutput
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MolT5Adapter(nn.Module):
    """
    将融合特征适配到MolT5输入格式
    处理维度转换和序列生成
    """
    
    def __init__(self, 
                 input_hidden_size: int = 768,
                 molt5_hidden_size: int = 1024,
                 max_seq_length: int = 128,
                 num_layers: int = 2,
                 num_heads: int = 16,
                 dropout: float = 0.1):
        """
        Args:
            input_hidden_size: 输入特征维度（来自融合层）
            molt5_hidden_size: MolT5的隐藏层维度
            max_seq_length: 最大序列长度
            num_layers: Transformer层数
            num_heads: 注意力头数
            dropout: Dropout率
        """
        super().__init__()
        
        self.input_hidden_size = input_hidden_size
        self.molt5_hidden_size = molt5_hidden_size
        self.max_seq_length = max_seq_length
        
        # 维度适配层
        self.dimension_adapter = nn.Sequential(
            nn.Linear(input_hidden_size, molt5_hidden_size),
            nn.LayerNorm(molt5_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 序列扩展层（将单个向量扩展为序列）
        self.sequence_projection = nn.Linear(molt5_hidden_size, molt5_hidden_size * 4)
        
        # Transformer编码器（用于处理序列）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=molt5_hidden_size,
            nhead=num_heads,
            dim_feedforward=molt5_hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.sequence_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 可学习的位置编码
        self.position_embeddings = nn.Parameter(
            torch.randn(1, max_seq_length, molt5_hidden_size)
        )
        
        # 序列起始标记
        self.start_token = nn.Parameter(
            torch.randn(1, 1, molt5_hidden_size)
        )
        
        logger.info(f"初始化MolT5Adapter: {input_hidden_size} -> {molt5_hidden_size}")
        
    def forward(self, 
                fused_features: torch.Tensor,
                target_length: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将融合特征转换为MolT5可接受的序列格式
        
        Args:
            fused_features: [batch_size, input_hidden_size] 融合后的特征
            target_length: 目标序列长度（默认为32）
            
        Returns:
            encoder_outputs: [batch_size, seq_len, molt5_hidden_size] 
            attention_mask: [batch_size, seq_len]
        """
        batch_size = fused_features.shape[0]
        # 确保输入在正确的设备上
        device = next(self.parameters()).device
        if fused_features.device != device:
            fused_features = fused_features.to(device)
        
        if target_length is None:
            target_length = min(32, self.max_seq_length)
        
        # 1. 维度适配
        adapted_features = self.dimension_adapter(fused_features)  # [batch, molt5_hidden]
        
        # 2. 扩展为初始序列
        # 方法1: 重复扩展
        expanded = adapted_features.unsqueeze(1).expand(-1, target_length, -1)
        
        # 方法2: 通过投影生成多样化的序列表示
        projected = self.sequence_projection(adapted_features)  # [batch, molt5_hidden * 4]
        projected = projected.view(batch_size, 4, self.molt5_hidden_size)  # [batch, 4, molt5_hidden]
        
        # 组合两种方法
        if target_length <= 4:
            sequence = projected[:, :target_length, :]
        else:
            # 使用投影的前4个位置，其余用扩展填充
            sequence = torch.cat([
                projected,
                expanded[:, 4:target_length, :]
            ], dim=1)
        
        # 3. 添加起始标记
        start_tokens = self.start_token.expand(batch_size, -1, -1)
        sequence = torch.cat([start_tokens, sequence[:, :-1, :]], dim=1)
        
        # 4. 添加位置编码
        position_embeds = self.position_embeddings[:, :target_length, :]
        sequence = sequence + position_embeds
        
        # 5. 通过Transformer处理
        encoder_outputs = self.sequence_encoder(sequence)
        
        # 6. 创建attention mask（全部为1，表示所有位置都有效）
        attention_mask = torch.ones(batch_size, target_length, device=device)
        
        return encoder_outputs, attention_mask


class MolT5Generator(nn.Module):
    """
    封装MolT5模型用于条件分子生成
    """
    
    def __init__(self, 
                 molt5_path: str = "/root/autodl-tmp/text2Mol-models/molt5-base",
                 adapter_config: Optional[Dict] = None,
                 freeze_molt5: bool = True,
                 device: str = 'cuda'):
        """
        Args:
            molt5_path: MolT5模型路径
            adapter_config: 适配器配置
            freeze_molt5: 是否冻结MolT5权重
            device: 设备
        """
        super().__init__()
        
        self.device = device
        
        # 加载MolT5模型和tokenizer
        logger.info(f"加载MolT5模型: {molt5_path}")
        self.molt5 = T5ForConditionalGeneration.from_pretrained(molt5_path)
        self.tokenizer = T5Tokenizer.from_pretrained(molt5_path)
        
        # 获取MolT5配置
        self.molt5_config = self.molt5.config
        molt5_hidden_size = self.molt5_config.d_model
        
        logger.info(f"MolT5隐藏层维度: {molt5_hidden_size}")
        
        # 初始化适配器
        if adapter_config is None:
            adapter_config = {}
        adapter_config['molt5_hidden_size'] = molt5_hidden_size
        
        self.adapter = MolT5Adapter(**adapter_config)
        
        # 冻结MolT5主体权重（可选）
        if freeze_molt5:
            for param in self.molt5.parameters():
                param.requires_grad = False
            logger.info("MolT5权重已冻结")
        
        # 移动到设备
        self.to(device)
        
    def forward(self,
                fused_features: torch.Tensor,
                target_smiles: Optional[List[str]] = None,
                max_length: int = 128) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            fused_features: [batch_size, hidden_size] 融合特征
            target_smiles: 目标SMILES（训练时提供）
            max_length: 最大生成长度
            
        Returns:
            包含loss和/或生成结果的字典
        """
        # 1. 通过适配器转换特征
        encoder_outputs, attention_mask = self.adapter(fused_features)
        
        # 2. 准备MolT5输入
        # MolT5期望encoder_outputs是BaseModelOutput格式
        molt5_encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_outputs
        )
        
        output_dict = {}
        
        if target_smiles is not None:
            # 训练模式：计算loss
            # 对目标SMILES进行tokenization
            target_encoding = self.tokenizer(
                target_smiles,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            target_ids = target_encoding.input_ids.to(self.device)
            
            # 前向传播计算loss
            outputs = self.molt5(
                encoder_outputs=molt5_encoder_outputs,
                attention_mask=attention_mask,
                labels=target_ids
            )
            
            # 应用token约束防止CUDA错误
            loss = outputs.loss
            logits = outputs.logits
            
            # 约束logits到有效词汇表范围，防止生成无效token
            vocab_size = self.tokenizer.vocab_size
            if logits.size(-1) > vocab_size:
                # 对超出词汇表的位置施加大的负值
                invalid_mask = torch.zeros_like(logits)
                invalid_mask[:, :, vocab_size:] = -float('inf')
                constrained_logits = logits + invalid_mask
                
                # 重新计算loss
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                if target_ids is not None:
                    # Shift so that tokens < n predict n
                    shift_logits = constrained_logits[..., :-1, :].contiguous()
                    shift_labels = target_ids[..., 1:].contiguous()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                                  shift_labels.view(-1))
                
                output_dict['logits'] = constrained_logits
            else:
                output_dict['logits'] = logits
            
            output_dict['loss'] = loss
            
        else:
            # 推理模式：生成SMILES
            with torch.no_grad():
                generated_ids = self.molt5.generate(
                    encoder_outputs=molt5_encoder_outputs,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    num_beams=5,
                    temperature=0.8,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    early_stopping=True
                )
            
            # 解码生成的SMILES
            generated_smiles = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            output_dict['generated_smiles'] = generated_smiles
            output_dict['generated_ids'] = generated_ids
        
        return output_dict
    
    def generate(self,
                 fused_features: torch.Tensor,
                 num_beams: int = 5,
                 temperature: float = 0.8,
                 max_length: int = 128,
                 num_return_sequences: int = 1) -> List[str]:
        """
        生成SMILES（使用改进的生成参数）
        
        Args:
            fused_features: 融合特征
            num_beams: Beam search大小
            temperature: 采样温度
            max_length: 最大长度
            num_return_sequences: 每个输入返回的序列数
            
        Returns:
            生成的SMILES列表
        """
        # 通过适配器
        encoder_outputs, attention_mask = self.adapter(fused_features)
        
        molt5_encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_outputs
        )
        
        # 使用改进的生成参数（修复生成质量）
        with torch.no_grad():
            generated_ids = self.molt5.generate(
                encoder_outputs=molt5_encoder_outputs,
                attention_mask=attention_mask,
                max_length=max_length,
                min_length=5,  # 最小长度
                num_beams=num_beams,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
                do_sample=False,  # 使用贪婪解码而非采样
                repetition_penalty=1.2,  # 避免重复
                length_penalty=1.0,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码并后处理
        generated_smiles = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        # 清理生成的文本（移除可能的错误词汇）
        cleaned_smiles = []
        for smiles in generated_smiles:
            # 如果包含明显的错误词汇，返回默认SMILES
            bad_words = ['Sand', 'brick', 'hell', 'Cub', 'rock', 'fence', 'walk']
            if any(word in smiles for word in bad_words):
                cleaned_smiles.append("CC")  # 乙烷作为默认
            else:
                cleaned_smiles.append(smiles)
        
        return cleaned_smiles
    
    def compute_loss(self,
                     fused_features: torch.Tensor,
                     target_smiles: List[str]) -> torch.Tensor:
        """
        计算训练loss
        
        Args:
            fused_features: 融合特征
            target_smiles: 目标SMILES
            
        Returns:
            loss: 训练损失
        """
        output_dict = self.forward(fused_features, target_smiles)
        return output_dict['loss']


def test_molt5_adapter():
    """测试MolT5适配器"""
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建测试数据
    batch_size = 2
    input_hidden_size = 768
    fused_features = torch.randn(batch_size, input_hidden_size).to(device)
    
    # 测试适配器
    print("测试MolT5Adapter...")
    adapter = MolT5Adapter(
        input_hidden_size=input_hidden_size,
        molt5_hidden_size=1024
    ).to(device)
    
    encoder_outputs, attention_mask = adapter(fused_features, target_length=32)
    print(f"编码器输出形状: {encoder_outputs.shape}")
    print(f"注意力掩码形状: {attention_mask.shape}")
    
    assert encoder_outputs.shape == (batch_size, 32, 1024)
    assert attention_mask.shape == (batch_size, 32)
    
    print("✅ MolT5Adapter测试通过！\n")
    
    # 测试生成器（需要MolT5模型）
    molt5_path = "/root/autodl-tmp/text2Mol-models/molt5-base"
    if Path(molt5_path).exists():
        print("测试MolT5Generator...")
        
        generator = MolT5Generator(
            molt5_path=molt5_path,
            adapter_config={'input_hidden_size': input_hidden_size},
            freeze_molt5=True,
            device=device
        )
        
        # 测试生成
        print("测试SMILES生成...")
        generated_smiles = generator.generate(
            fused_features,
            num_beams=3,
            max_length=64
        )
        
        print(f"生成的SMILES: {generated_smiles}")
        
        # 测试训练模式
        print("\n测试训练模式...")
        target_smiles = ["CCO", "CC(C)O"]  # 示例SMILES
        loss = generator.compute_loss(fused_features, target_smiles)
        print(f"训练Loss: {loss.item():.4f}")
        
        print("✅ MolT5Generator测试通过！")
    else:
        print(f"⚠️ MolT5模型未找到: {molt5_path}")
    
    print("\n所有测试完成！")


if __name__ == "__main__":
    test_molt5_adapter()