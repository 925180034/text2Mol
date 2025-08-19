#!/usr/bin/env python3
"""
详细测试Scaffold + Text -> SMILES生成
展示模型的实际输入输出
"""

import torch
import pandas as pd
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/root/text2Mol/scaffold-mol-generation')
from scaffold_mol_gen.models.end2end_model import End2EndMolecularGenerator
from rdkit import Chem

def load_model():
    """加载训练好的Stage 2模型"""
    print("加载模型...")
    model = End2EndMolecularGenerator(
        hidden_size=768,
        molt5_path='/root/autodl-tmp/text2Mol-models/molt5-base',
        use_scibert=False,
        freeze_encoders=False,
        freeze_molt5=False,
        fusion_type='both',
        device='cuda'
    )
    
    checkpoint = torch.load(
        "/root/autodl-tmp/text2Mol-stage2/best_model_stage2.pt",
        map_location='cuda',
        weights_only=False
    )
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
    
    model.eval()
    model.cuda()
    return model

def test_generation():
    """测试生成过程，展示详细的输入输出"""
    
    # 加载模型
    model = load_model()
    print("✅ 模型加载成功\n")
    
    # 加载测试数据
    test_df = pd.read_csv('/root/text2Mol/scaffold-mol-generation/Datasets/test_with_scaffold.csv')
    
    # 测试前3个样本
    print("="*80)
    print("🧪 详细测试: Scaffold + Text -> SMILES")
    print("="*80)
    
    for i in range(3):
        print(f"\n📝 样本 {i+1}:")
        print("-"*60)
        
        # 获取输入
        scaffold = test_df.iloc[i]['scaffold']
        text = test_df.iloc[i]['description']
        target = test_df.iloc[i]['SMILES']
        
        # 显示输入
        print(f"🔹 输入Scaffold (SMILES): {scaffold[:50]}...")
        print(f"🔹 输入Text描述: {text[:100]}...")
        print(f"🔹 目标SMILES: {target[:50]}...")
        
        # 验证scaffold是否有效
        scaffold_mol = Chem.MolFromSmiles(scaffold)
        if scaffold_mol:
            print(f"   ✓ Scaffold是有效的SMILES (原子数: {scaffold_mol.GetNumAtoms()})")
        else:
            print(f"   ✗ Scaffold无效")
        
        # 生成分子
        with torch.no_grad():
            try:
                generated = model.generate(
                    [scaffold],  # Scaffold输入
                    [text],      # Text输入
                    scaffold_modality='smiles',
                    max_length=256,
                    num_beams=5
                )[0]
                
                print(f"\n🔸 生成的SMILES: {generated[:100]}...")
                
                # 检查生成的SMILES是否有效
                gen_mol = Chem.MolFromSmiles(generated)
                if gen_mol:
                    print(f"   ✓ 生成了有效的SMILES (原子数: {gen_mol.GetNumAtoms()})")
                    
                    # 计算相似度
                    from rdkit.Chem import AllChem, DataStructs
                    target_mol = Chem.MolFromSmiles(target)
                    if target_mol:
                        fp1 = AllChem.GetMorganFingerprintAsBitVect(gen_mol, 2)
                        fp2 = AllChem.GetMorganFingerprintAsBitVect(target_mol, 2)
                        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
                        print(f"   📊 与目标的Tanimoto相似度: {similarity:.4f}")
                else:
                    print(f"   ✗ 生成的SMILES无效")
                    
            except Exception as e:
                print(f"   ❌ 生成错误: {e}")
        
        print("-"*60)
    
    print("\n" + "="*80)
    print("💡 分析:")
    print("- 模型接收了正确的Scaffold (SMILES格式) + Text输入")
    print("- 但生成的分子质量很差，主要是重复的碳链")
    print("- 模型没有学会如何根据Scaffold骨架和Text描述生成正确的分子")
    print("="*80)

if __name__ == "__main__":
    test_generation()