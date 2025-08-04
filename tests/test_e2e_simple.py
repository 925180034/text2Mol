"""
简化的端到端测试脚本
修复设备不匹配问题
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_simple_pipeline():
    """测试简化的端到端流程"""
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 导入模块
    from scaffold_mol_gen.models.encoders import MultiModalEncoder
    from scaffold_mol_gen.models.fusion_simplified import ModalFusionLayer
    from scaffold_mol_gen.models.molt5_adapter import MolT5Generator
    
    print("\n1. 初始化组件...")
    
    # 创建编码器
    encoder = MultiModalEncoder(
        hidden_size=768,
        use_scibert=False,
        freeze_backbones=True,
        device=device
    ).to(device)
    
    # 创建融合层
    fusion = ModalFusionLayer(
        hidden_size=768,
        fusion_type='both'
    ).to(device)
    
    # 创建MolT5生成器
    generator = MolT5Generator(
        molt5_path="/root/autodl-tmp/text2Mol-models/MolT5-Large-Caption2SMILES",
        adapter_config={'input_hidden_size': 768},
        freeze_molt5=True,
        device=device
    )
    
    print("✅ 组件初始化完成")
    
    # 准备测试数据
    print("\n2. 准备测试数据...")
    scaffold_smiles = "c1ccccc1"  # 苯环
    text_description = "Anti-inflammatory drug with carboxylic acid"
    
    print(f"Scaffold: {scaffold_smiles}")
    print(f"Text: {text_description}")
    
    # 测试流程
    print("\n3. 执行端到端流程...")
    
    try:
        # Step 1: 编码
        print("   - 编码输入...")
        with torch.no_grad():
            scaffold_features, text_features = encoder(
                scaffold_data=scaffold_smiles,
                text_data=text_description,
                scaffold_modality='smiles'
            )
        
        # 确保在正确的设备上
        scaffold_features = scaffold_features.to(device)
        text_features = text_features.to(device)
        
        print(f"   Scaffold特征: {scaffold_features.shape}")
        print(f"   Text特征: {text_features.shape}")
        
        # Step 2: 融合
        print("   - 融合特征...")
        with torch.no_grad():
            fused_features, fusion_info = fusion(
                scaffold_features=scaffold_features,
                text_features=text_features
            )
        
        fused_features = fused_features.to(device)
        print(f"   融合特征: {fused_features.shape}")
        print(f"   门控权重 - Scaffold: {fusion_info.get('scaffold_weight', 0):.2%}, Text: {fusion_info.get('text_weight', 0):.2%}")
        
        # Step 3: 生成
        print("   - 生成SMILES...")
        generated_smiles = generator.generate(
            fused_features=fused_features,
            num_beams=3,
            temperature=0.8,
            max_length=64
        )
        
        print(f"\n✅ 生成成功!")
        print(f"生成的SMILES: {generated_smiles[0]}")
        
        # 验证SMILES
        from rdkit import Chem
        mol = Chem.MolFromSmiles(generated_smiles[0])
        if mol is not None:
            print("✅ SMILES有效")
            print(f"   分子式: {Chem.rdMolDescriptors.CalcMolFormula(mol)}")
            print(f"   分子量: {Chem.rdMolDescriptors.CalcExactMolWt(mol):.2f}")
        else:
            print("⚠️ SMILES无效（这在初期训练前是正常的）")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试批处理
    print("\n4. 测试批处理...")
    
    try:
        batch_scaffold = ["c1ccccc1", "c1ccc2c(c1)cccc2"]  # 苯环和萘环
        batch_text = ["Anti-inflammatory", "Antibiotic"]
        
        with torch.no_grad():
            # 编码
            scaffold_features, text_features = encoder(
                scaffold_data=batch_scaffold,
                text_data=batch_text,
                scaffold_modality='smiles'
            )
            
            scaffold_features = scaffold_features.to(device)
            text_features = text_features.to(device)
            
            # 融合
            fused_features, _ = fusion(
                scaffold_features=scaffold_features,
                text_features=text_features
            )
            
            fused_features = fused_features.to(device)
            
            # 生成
            generated_batch = generator.generate(
                fused_features=fused_features,
                num_beams=2,
                max_length=64
            )
        
        print(f"✅ 批处理成功!")
        for i, smiles in enumerate(generated_batch):
            print(f"   样本{i+1}: {smiles[:50]}...")
            
    except Exception as e:
        print(f"❌ 批处理错误: {e}")
    
    # 测试训练模式
    print("\n5. 测试训练模式...")
    
    try:
        # 准备目标SMILES
        target_smiles = ["CC(=O)Oc1ccccc1C(=O)O", "CC1CC(C)CN1C(=O)C"]
        
        # 计算loss
        loss = generator.compute_loss(
            fused_features=fused_features,
            target_smiles=target_smiles
        )
        
        print(f"✅ 训练Loss: {loss.item():.4f}")
        
    except Exception as e:
        print(f"❌ 训练模式错误: {e}")
    
    print("\n" + "=" * 50)
    print("测试完成！")
    print("=" * 50)
    
    # 显示模型统计
    print("\n模型统计:")
    total_params = sum(p.numel() for p in encoder.parameters()) + \
                   sum(p.numel() for p in fusion.parameters()) + \
                   sum(p.numel() for p in generator.adapter.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad) + \
                       sum(p.numel() for p in fusion.parameters() if p.requires_grad) + \
                       sum(p.numel() for p in generator.adapter.parameters() if p.requires_grad)
    
    print(f"总参数量: {total_params/1e6:.2f}M")
    print(f"可训练参数: {trainable_params/1e6:.2f}M")
    print(f"冻结参数: {(total_params-trainable_params)/1e6:.2f}M")


if __name__ == "__main__":
    test_simple_pipeline()