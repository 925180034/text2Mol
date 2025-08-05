#!/usr/bin/env python3
"""
分子生成推理脚本 - 支持3种输入模态生成完整分子
"""

import argparse
import torch
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

from scaffold_mol_gen.models.stable_model import StableMolecularGenerator
from scaffold_mol_gen.data.multimodal_preprocessor import MultiModalPreprocessor
from transformers import AutoTokenizer

class MolecularInference:
    def __init__(self, model_path, device='cuda'):
        """初始化推理引擎"""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载模型
        print("加载模型...")
        self.model = self._load_model(model_path)
        
        # 初始化tokenizers
        self.smiles_tokenizer = AutoTokenizer.from_pretrained('laituan245/molt5-large-smiles2caption', model_max_length=512)
        self.text_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        
        # 模态转换器
        self.converter = MultiModalPreprocessor()
        
    def _load_model(self, model_path):
        """加载训练好的模型"""
        # 创建模型
        model = StableMolecularGenerator(
            freeze_molt5_encoder=True,
            freeze_molt5_decoder_layers=16,
            use_simple_fusion=True
        ).to(self.device)
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"模型加载成功！训练轮数: {checkpoint.get('epoch', 'unknown')}")
        print(f"最佳验证损失: {checkpoint.get('best_loss', 'unknown')}")
        
        return model
    
    def generate_molecule(self, scaffold, text_description, scaffold_type='smiles', 
                         num_beams=5, max_length=512, temperature=1.0):
        """
        生成分子
        
        Args:
            scaffold: 分子骨架 (SMILES字符串、图像路径或已处理的图数据)
            text_description: 文本描述
            scaffold_type: 骨架输入类型 ('smiles', 'image', 'graph')
            num_beams: beam search的beam数量
            max_length: 最大生成长度
            temperature: 生成温度
        """
        # 1. 处理文本输入
        text_inputs = self.text_tokenizer(
            text_description,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # 2. 处理骨架输入（根据类型）
        if scaffold_type == 'smiles':
            scaffold_inputs = self.smiles_tokenizer(
                scaffold,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
        elif scaffold_type == 'image':
            # 图像输入暂不支持
            raise NotImplementedError("图像输入暂时不支持，请使用SMILES格式")
        elif scaffold_type == 'graph':
            # 图数据输入暂不支持
            raise NotImplementedError("图数据输入暂时不支持，请使用SMILES格式")
        else:
            raise ValueError(f"不支持的骨架类型: {scaffold_type}")
        
        # 3. 生成分子
        with torch.no_grad():
            # 准备输入
            batch = {
                'text_input_ids': text_inputs['input_ids'],
                'text_attention_mask': text_inputs['attention_mask'],
                'scaffold_input_ids': scaffold_inputs['input_ids'],
                'scaffold_attention_mask': scaffold_inputs['attention_mask']
            }
            
            # 生成
            outputs = self.model.generate(
                **batch,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                do_sample=temperature > 0,
                top_k=50,
                top_p=0.95,
                early_stopping=True
            )
            
            # 解码
            generated_smiles = self.smiles_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        return generated_smiles
    
    def visualize_result(self, scaffold_smiles, generated_smiles, save_path=None):
        """可视化骨架和生成的分子"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 绘制骨架
        scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)
        if scaffold_mol:
            img1 = Draw.MolToImage(scaffold_mol)
            ax1.imshow(img1)
            ax1.set_title("输入骨架")
            ax1.axis('off')
        
        # 绘制生成的分子
        generated_mol = Chem.MolFromSmiles(generated_smiles)
        if generated_mol:
            img2 = Draw.MolToImage(generated_mol)
            ax2.imshow(img2)
            ax2.set_title("生成的完整分子")
            ax2.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"结果已保存到: {save_path}")
        else:
            plt.show()
        
        return fig

def main():
    parser = argparse.ArgumentParser(description='分子生成推理')
    parser.add_argument('--model', type=str, default='/root/autodl-tmp/safe_fast_checkpoints/best_model.pt',
                       help='模型路径')
    parser.add_argument('--scaffold', type=str, required=True,
                       help='分子骨架 (SMILES或图像路径)')
    parser.add_argument('--text', type=str, required=True,
                       help='文本描述')
    parser.add_argument('--type', type=str, default='smiles',
                       choices=['smiles', 'image', 'graph'],
                       help='骨架输入类型')
    parser.add_argument('--beams', type=int, default=5,
                       help='Beam search大小')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='生成温度')
    parser.add_argument('--output', type=str, default='generated_molecule.png',
                       help='输出图像路径')
    
    args = parser.parse_args()
    
    # 创建推理引擎
    inference = MolecularInference(args.model)
    
    # 生成分子
    print(f"\n输入骨架: {args.scaffold}")
    print(f"文本描述: {args.text}")
    print(f"骨架类型: {args.type}")
    print("\n生成中...")
    
    generated_smiles = inference.generate_molecule(
        scaffold=args.scaffold,
        text_description=args.text,
        scaffold_type=args.type,
        num_beams=args.beams,
        temperature=args.temperature
    )
    
    print(f"\n生成的分子SMILES: {generated_smiles}")
    
    # 验证生成的分子
    mol = Chem.MolFromSmiles(generated_smiles)
    if mol:
        print("✅ 生成的分子有效!")
        print(f"分子量: {Chem.Descriptors.MolWt(mol):.2f}")
        print(f"原子数: {mol.GetNumAtoms()}")
        
        # 可视化
        if args.type == 'smiles':
            inference.visualize_result(args.scaffold, generated_smiles, args.output)
    else:
        print("❌ 生成的分子无效")

if __name__ == '__main__':
    main()