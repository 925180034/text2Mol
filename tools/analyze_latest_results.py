#!/usr/bin/env python3
"""
分析最新的训练结果
判断效果和是否需要继续训练其他模态
"""

import pandas as pd
from pathlib import Path

def analyze_latest_results():
    print("📊 分析最新训练结果")
    print("=" * 60)
    
    # 检查最新实验目录
    latest_dir = "experiments/demo_multimodal_20250805_184448"
    
    if not Path(latest_dir).exists():
        print(f"❌ 未找到最新实验目录: {latest_dir}")
        return
    
    print(f"✅ 找到最新实验结果: {latest_dir}")
    
    # 分析各模态结果
    modalities = ['smiles', 'graph', 'image']
    results = {}
    
    for modality in modalities:
        csv_file = f"{latest_dir}/{modality}/inference_results.csv"
        if Path(csv_file).exists():
            df = pd.read_csv(csv_file)
            
            # 计算指标
            total_samples = len(df)
            valid_samples = df['valid'].sum()
            validity = valid_samples / total_samples * 100 if total_samples > 0 else 0
            
            results[modality] = {
                'total': total_samples,
                'valid': valid_samples,
                'validity': validity,
                'data': df
            }
            
            print(f"✅ {modality.upper()}模态: {valid_samples}/{total_samples} 有效 ({validity:.1f}%)")
        else:
            print(f"❌ 未找到{modality}模态结果文件")
    
    # 深度分析
    print(f"\n🔍 深度分析:")
    print("-" * 40)
    
    # 1. 检查模态差异
    if len(results) >= 2:
        validities = [results[m]['validity'] for m in results.keys()]
        max_validity = max(validities)
        min_validity = min(validities)
        diff = max_validity - min_validity
        
        print(f"模态间有效性差异: {diff:.1f}%")
        
        if diff < 10:
            print("⚠️ 模态间差异很小，可能使用了相同的底层模型")
        else:
            print("✅ 模态间有明显差异，说明真正学会了多模态处理")
    
    # 2. 检查生成质量
    if 'smiles' in results:
        smiles_data = results['smiles']['data']
        
        # 分析生成的SMILES
        generated_lengths = smiles_data['generated'].str.len()
        avg_length = generated_lengths.mean()
        
        print(f"生成SMILES平均长度: {avg_length:.1f}")
        
        # 检查是否有重复生成
        unique_generated = smiles_data['generated'].nunique()
        total_generated = len(smiles_data)
        uniqueness = unique_generated / total_generated * 100 if total_generated > 0 else 0
        
        print(f"生成唯一性: {uniqueness:.1f}%")
        
        # 显示几个成功示例
        valid_samples = smiles_data[smiles_data['valid'] == True]
        if len(valid_samples) > 0:
            print(f"\n✅ 成功生成示例:")
            for i, row in valid_samples.head(3).iterrows():
                target = row['target'][:50] + "..." if len(row['target']) > 50 else row['target']
                generated = row['generated'][:50] + "..." if len(row['generated']) > 50 else row['generated']
                print(f"  样本{i}: 目标 → 生成")
                print(f"    {target}")
                print(f"    {generated}")
    
    # 3. 训练效果评估
    print(f"\n🎯 训练效果评估:")
    print("-" * 40)
    
    if 'smiles' in results:
        smiles_validity = results['smiles']['validity']
        
        if smiles_validity >= 70:
            print("🎉 优秀！SMILES模态训练效果很好")
            verdict = "excellent"
        elif smiles_validity >= 50:
            print("✅ 良好！SMILES模态有基本的生成能力")  
            verdict = "good"
        elif smiles_validity >= 30:
            print("⚠️ 一般。SMILES模态需要改进")
            verdict = "fair"
        else:
            print("❌ 较差。SMILES模态需要重新训练")
            verdict = "poor"
    
    # 4. 多模态训练建议
    print(f"\n💡 多模态训练建议:")
    print("-" * 40)
    
    if len(results) == 3:  # 有三个模态结果
        all_validities = [results[m]['validity'] for m in results.keys()]
        
        if max(all_validities) - min(all_validities) < 5:
            print("⚠️ 当前结果可能是用同一个SMILES模型测试的")
            print("📋 建议：需要分别训练Graph和Image模态")
            print("   1. python safe_background_training.py graph")
            print("   2. python safe_background_training.py image")
            need_training = True
        else:
            print("✅ 三个模态都已独立训练，有明显差异")
            print("🎉 多模态训练已完成！")
            need_training = False
    else:
        print("❌ 缺少某些模态的结果")
        need_training = True
    
    # 5. 后台运行能力
    print(f"\n🔄 关于后台运行:")
    print("-" * 40)
    print("✅ 安全训练脚本支持完全后台运行")
    print("✅ 包含磁盘空间监控和自动保护")
    print("✅ 可以同时运行训练和监控")
    print()
    print("启动方式:")
    print("  终端1: python safe_background_training.py graph")  
    print("  终端2: python training_monitor.py")
    
    # 总结建议
    print(f"\n🏆 总结建议:")
    print("=" * 60)
    
    if 'smiles' in results and results['smiles']['validity'] >= 50:
        print("✅ SMILES模态训练成功，有基本的分子生成能力")
        
        if need_training:
            print("🔄 下一步：继续训练Graph和Image模态以获得完整多模态能力")
            print("⚡ 推荐使用安全训练脚本，避免磁盘空间问题")
            print()
            print("立即开始：")
            print("  python safe_background_training.py graph")
        else:
            print("🎉 多模态训练已完成！可以开始应用")
    else:
        if 'smiles' in results:
            print("⚠️ SMILES模态效果不够理想，建议重新训练")
        else:
            print("❌ 缺少SMILES模态结果")

if __name__ == "__main__":
    analyze_latest_results()