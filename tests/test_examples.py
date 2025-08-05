#!/usr/bin/env python3
"""
快速测试脚本 - 用几个示例测试模型
"""

from inference import MolecularInference
import pandas as pd

# 测试示例
test_examples = [
    {
        "scaffold": "c1ccccc1",  # 苯环骨架
        "text": "A molecule with anti-inflammatory properties and good oral bioavailability"
    },
    {
        "scaffold": "C1CCCCC1",  # 环己烷骨架
        "text": "A compound with analgesic effects and low toxicity"
    },
    {
        "scaffold": "c1ccc2c(c1)ncc2",  # 喹啉骨架
        "text": "An antimalarial agent with improved resistance profile"
    }
]

def main():
    # 初始化推理引擎
    print("初始化模型...")
    model_path = '/root/autodl-tmp/safe_fast_checkpoints/best_model.pt'
    inference = MolecularInference(model_path)
    
    print("\n=== 测试示例 ===")
    
    for i, example in enumerate(test_examples, 1):
        print(f"\n示例 {i}:")
        print(f"骨架: {example['scaffold']}")
        print(f"描述: {example['text']}")
        
        try:
            # 生成分子
            generated = inference.generate_molecule(
                scaffold=example['scaffold'],
                text_description=example['text'],
                scaffold_type='smiles',
                num_beams=5,
                temperature=1.0
            )
            
            print(f"生成: {generated}")
            
            # 保存可视化
            output_path = f"example_{i}_result.png"
            inference.visualize_result(
                example['scaffold'], 
                generated, 
                save_path=output_path
            )
            
        except Exception as e:
            print(f"生成失败: {e}")
    
    print("\n=== 从测试集中随机选择几个例子 ===")
    
    # 读取测试集
    test_df = pd.read_csv('Datasets/test.csv')
    
    # 随机选择3个样本
    random_samples = test_df.sample(n=3, random_state=42)
    
    for idx, row in random_samples.iterrows():
        print(f"\n测试集样本:")
        print(f"骨架: {row['scaffold']}")
        print(f"描述: {row['text'][:100]}...")
        print(f"真实: {row['SMILES']}")
        
        try:
            generated = inference.generate_molecule(
                scaffold=row['scaffold'],
                text_description=row['text'],
                scaffold_type='smiles'
            )
            print(f"生成: {generated}")
            
        except Exception as e:
            print(f"生成失败: {e}")

if __name__ == '__main__':
    main()