#!/usr/bin/env python3
"""
训练MolT5-base模型生成SMILES
从头开始训练，让模型学会生成分子SMILES
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, get_linear_schedule_with_warmup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleSMILESDataset(Dataset):
    """简单的SMILES生成数据集"""
    
    def __init__(self, csv_file, tokenizer, max_length=128):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 只保留有效的SMILES
        from rdkit import Chem
        valid_mask = self.data['SMILES'].apply(lambda x: Chem.MolFromSmiles(x) is not None)
        self.data = self.data[valid_mask].reset_index(drop=True)
        
        logger.info(f"加载了 {len(self.data)} 个有效的SMILES")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 输入：文本描述
        text = row.get('description', row.get('text', ''))
        if pd.isna(text):
            text = "Generate a molecule"
        
        # 添加任务前缀（帮助模型理解任务）
        input_text = f"Generate SMILES: {text}"
        
        # 目标：SMILES
        target_smiles = row['SMILES']
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        targets = self.tokenizer(
            target_smiles,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 将padding位置设为-100（忽略loss）
        labels = targets['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels
        }

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        # 移动到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # 更新进度条
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, tokenizer, device, num_samples=5):
    """评估模型"""
    model.eval()
    total_loss = 0
    generated_samples = []
    target_samples = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 计算loss
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            total_loss += outputs.loss.item()
            
            # 生成样本
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=5,
                do_sample=False,
                early_stopping=True
            )
            
            # 解码
            for gen_id, label in zip(generated_ids, labels):
                gen_text = tokenizer.decode(gen_id, skip_special_tokens=True)
                # 清理标签中的-100
                label[label == -100] = tokenizer.pad_token_id
                target_text = tokenizer.decode(label, skip_special_tokens=True)
                
                generated_samples.append(gen_text)
                target_samples.append(target_text)
    
    # 计算SMILES有效率
    from rdkit import Chem
    valid_count = sum(1 for s in generated_samples if Chem.MolFromSmiles(s) is not None)
    validity = valid_count / len(generated_samples) if generated_samples else 0
    
    return total_loss / min(len(dataloader), num_samples), validity, generated_samples, target_samples

def main():
    # 设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    learning_rate = 5e-5
    num_epochs = 10
    warmup_steps = 500
    
    # 输出目录
    output_dir = f"/root/autodl-tmp/text2Mol-outputs/molt5_smiles_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("🚀 训练MolT5生成SMILES")
    logger.info("="*70)
    logger.info(f"设备: {device}")
    logger.info(f"输出目录: {output_dir}")
    
    # 加载模型和tokenizer
    model_path = "/root/autodl-tmp/text2Mol-models/molt5-base"
    logger.info(f"加载模型: {model_path}")
    
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    
    model.to(device)
    
    # 准备数据
    logger.info("准备数据集...")
    train_dataset = SimpleSMILESDataset(
        "Datasets/train.csv",
        tokenizer
    )
    
    val_dataset = SimpleSMILESDataset(
        "Datasets/validation.csv",
        tokenizer
    )
    
    # 限制数据集大小以加快训练
    train_dataset.data = train_dataset.data[:5000]
    val_dataset.data = val_dataset.data[:500]
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"训练集: {len(train_dataset)} 样本")
    logger.info(f"验证集: {len(val_dataset)} 样本")
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 训练循环
    best_validity = 0
    
    for epoch in range(num_epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"{'='*50}")
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        logger.info(f"训练损失: {train_loss:.4f}")
        
        # 评估
        val_loss, validity, gen_samples, target_samples = evaluate(
            model, val_loader, tokenizer, device
        )
        
        logger.info(f"验证损失: {val_loss:.4f}")
        logger.info(f"SMILES有效率: {validity:.2%}")
        
        # 打印样本
        logger.info("\n生成样本:")
        for i in range(min(3, len(gen_samples))):
            logger.info(f"  目标: {target_samples[i]}")
            logger.info(f"  生成: {gen_samples[i]}")
            logger.info("")
        
        # 保存最佳模型
        if validity > best_validity:
            best_validity = validity
            logger.info(f"💾 保存最佳模型 (有效率: {validity:.2%})")
            
            # 保存模型
            model.save_pretrained(f"{output_dir}/best_model")
            tokenizer.save_pretrained(f"{output_dir}/best_model")
            
            # 保存checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_validity': best_validity,
            }, f"{output_dir}/best_checkpoint.pt")
    
    logger.info(f"\n{'='*70}")
    logger.info(f"训练完成！最佳有效率: {best_validity:.2%}")
    logger.info(f"模型保存在: {output_dir}")
    logger.info(f"{'='*70}")

if __name__ == "__main__":
    main()