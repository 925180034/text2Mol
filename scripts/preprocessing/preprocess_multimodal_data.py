#!/usr/bin/env python3
"""
多模态数据预处理脚本
将ChEBI-20数据集转换为支持7种输入输出组合的多模态格式
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from scaffold_mol_gen.data.multimodal_preprocessor import MultiModalPreprocessor

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='多模态数据预处理')
    
    parser.add_argument(
        '--input-dir',
        type=str,
        default='Datasets',
        help='输入数据集目录'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str,
        default='Datasets/multimodal',
        help='输出多模态数据目录'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=500,
        help='批处理大小'
    )
    
    parser.add_argument(
        '--image-size',
        type=int,
        nargs=2,
        default=[224, 224],
        help='分子图像尺寸 (高度 宽度)'
    )
    
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=['train.csv', 'validation.csv', 'test.csv'],
        help='要处理的数据集文件'
    )
    
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='仅运行测试，不处理完entire数据集'
    )
    
    args = parser.parse_args()
    
    logger.info("=== 多模态数据预处理开始 ===")
    logger.info(f"输入目录: {args.input_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"图像尺寸: {args.image_size}")
    logger.info(f"批处理大小: {args.batch_size}")
    
    # 初始化预处理器
    preprocessor = MultiModalPreprocessor(
        image_size=tuple(args.image_size),
        bond_features=True
    )
    
    # 测试模式
    if args.test_only:
        logger.info("运行测试模式...")
        test_preprocessor(preprocessor)
        return
    
    # 检查输入目录
    input_path = Path(args.input_dir)
    if not input_path.exists():
        logger.error(f"输入目录不存在: {input_path}")
        return
    
    # 创建输出目录
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 处理每个数据集
    total_statistics = {
        'total_datasets': 0,
        'successful_datasets': 0,
        'total_samples': 0,
        'successful_samples': 0,
        'dataset_results': {}
    }
    
    for dataset_file in args.datasets:
        dataset_path = input_path / dataset_file
        
        if not dataset_path.exists():
            logger.warning(f"数据集文件不存在: {dataset_path}")
            continue
        
        logger.info(f"\n处理数据集: {dataset_file}")
        
        # 创建数据集专用输出目录
        dataset_output = output_path / dataset_file.replace('.csv', '')
        
        try:
            # 处理数据集
            statistics = preprocessor.process_dataset(
                csv_path=str(dataset_path),
                output_dir=str(dataset_output),
                batch_size=args.batch_size
            )
            
            # 更新总统计
            total_statistics['total_datasets'] += 1
            total_statistics['successful_datasets'] += 1
            total_statistics['total_samples'] += statistics['total_samples']
            total_statistics['successful_samples'] += (
                statistics['successful_scaffold_graphs'] + 
                statistics['successful_scaffold_images'] +
                statistics['successful_target_graphs'] + 
                statistics['successful_target_images']
            )
            total_statistics['dataset_results'][dataset_file] = statistics
            
            logger.info(f"✅ {dataset_file} 处理完成")
            logger.info(f"   样本数: {statistics['total_samples']}")
            logger.info(f"   Scaffold图: {statistics['successful_scaffold_graphs']}")
            logger.info(f"   Scaffold图像: {statistics['successful_scaffold_images']}")
            logger.info(f"   Target图: {statistics['successful_target_graphs']}")
            logger.info(f"   Target图像: {statistics['successful_target_images']}")
            
        except Exception as e:
            logger.error(f"❌ 处理 {dataset_file} 失败: {e}")
            total_statistics['total_datasets'] += 1
            total_statistics['dataset_results'][dataset_file] = {'error': str(e)}
    
    # 保存总体统计信息
    import json
    with open(output_path / 'overall_statistics.json', 'w') as f:
        json.dump(total_statistics, f, indent=2)
    
    # 输出总结
    logger.info("\n=== 多模态数据预处理完成 ===")
    logger.info(f"处理数据集: {total_statistics['successful_datasets']}/{total_statistics['total_datasets']}")
    logger.info(f"总样本数: {total_statistics['total_samples']}")
    logger.info(f"成功转换: {total_statistics['successful_samples']}")
    logger.info(f"输出目录: {output_path}")
    
    # 创建7种组合的说明文档
    create_combination_docs(output_path)

def test_preprocessor(preprocessor: MultiModalPreprocessor):
    """测试预处理器功能"""
    logger.info("开始测试多模态预处理器...")
    
    # 测试样本
    test_samples = [
        "CCO",  # 乙醇
        "c1ccccc1",  # 苯
        "CC(=O)O",  # 乙酸
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # 咖啡因 (修正SMILES)
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # 布洛芬
    ]
    
    success_count = 0
    total_count = len(test_samples) * 2  # 每个样本测试Graph + Image
    
    for i, smiles in enumerate(test_samples):
        logger.info(f"测试样本 {i+1}: {smiles}")
        
        # 测试SMILES → Graph
        graph = preprocessor.smiles_to_graph(smiles)
        if graph is not None:
            logger.info(f"  ✅ Graph: {graph.x.shape[0]}个节点, {graph.edge_index.shape[1]}条边")
            success_count += 1
        else:
            logger.error(f"  ❌ Graph转换失败")
        
        # 测试SMILES → Image
        image = preprocessor.smiles_to_image(smiles)
        if image is not None:
            logger.info(f"  ✅ Image: {image.shape}")
            success_count += 1
        else:
            logger.error(f"  ❌ Image转换失败")
    
    success_rate = success_count / total_count * 100
    logger.info(f"\n测试完成: {success_count}/{total_count} ({success_rate:.1f}%) 成功")
    
    if success_rate >= 80:
        logger.info("✅ 预处理器测试通过!")
    else:
        logger.warning("⚠️ 预处理器测试存在问题，建议检查环境配置")

def create_combination_docs(output_path: Path):
    """创建7种输入输出组合的说明文档"""
    
    combinations_doc = """# 多模态分子生成 - 7种输入输出组合

## 数据格式说明

处理后的数据支持以下7种输入输出组合：

| # | Scaffold输入模态 | Text输入 | 输出模态 | 数据位置 |
|---|-----------------|----------|----------|----------|
| 1 | SMILES | ✓ | SMILES | 原始CSV + processed_dataset.csv |
| 2 | Image | ✓ | Image | images/scaffold_images.npy → images/target_images.npy |
| 3 | Image | ✓ | Graph | images/scaffold_images.npy → graphs/target_graphs.pt |
| 4 | Image | ✓ | SMILES | images/scaffold_images.npy → processed_dataset.csv |
| 5 | Graph | ✓ | Graph | graphs/scaffold_graphs.pt → graphs/target_graphs.pt |
| 6 | Graph | ✓ | Image | graphs/scaffold_graphs.pt → images/target_images.npy |
| 7 | Graph | ✓ | SMILES | graphs/scaffold_graphs.pt → processed_dataset.csv |

## 数据结构

### 图数据 (*.pt)
- PyTorch Geometric Data对象列表
- 节点特征: 原子特征 (原子序数、度数、电荷、杂化、芳香性、成环)
- 边特征: 键特征 (键类型、共轭、成环)

### 图像数据 (*.npy)
- NumPy数组列表
- 格式: RGB (H, W, 3)
- 尺寸: 224x224

### 文本数据
- 包含在processed_dataset.csv的'text'列中
- 分子性质和功能的自然语言描述

### 元数据 (metadata.json)
- 包含每个样本的原始索引、CID、SMILES等信息
- 用于数据追溯和质量控制

## 使用示例

```python
from scaffold_mol_gen.data.multimodal_preprocessor import MultiModalPreprocessor

# 加载预处理后的数据
preprocessor = MultiModalPreprocessor()
data = preprocessor.load_processed_data('Datasets/multimodal/train')

# 访问不同模态数据
scaffold_graphs = data['scaffold_graphs']
scaffold_images = data['scaffold_images'] 
target_graphs = data['target_graphs']
target_images = data['target_images']
metadata = data['metadata']
```

## 质量统计

详细的转换成功率和质量统计请查看各数据集目录下的 `statistics.json` 文件。
"""
    
    with open(output_path / 'README.md', 'w', encoding='utf-8') as f:
        f.write(combinations_doc)
    
    logger.info("✅ 创建组合说明文档完成")

if __name__ == "__main__":
    main()