#!/usr/bin/env python3
"""
系统状态检查脚本 - 验证环境和模型是否就绪
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """检查核心依赖"""
    logger.info("检查核心依赖...")
    
    try:
        import torch
        logger.info(f"✅ PyTorch: {torch.__version__}")
        logger.info(f"✅ CUDA可用: {torch.cuda.is_available()}")
        
        import torch_geometric
        logger.info(f"✅ PyTorch Geometric: {torch_geometric.__version__}")
        
        from rdkit import Chem
        logger.info("✅ RDKit: 可用")
        
        import transformers
        logger.info(f"✅ Transformers: {transformers.__version__}")
        
        return True
    except ImportError as e:
        logger.error(f"❌ 依赖缺失: {e}")
        return False

def check_model():
    """检查模型文件"""
    logger.info("检查模型文件...")
    
    model_path = Path("models/MolT5-Small")
    
    if not model_path.exists():
        logger.error("❌ 模型目录不存在")
        return False
    
    required_files = ["config.json", "model.safetensors", "tokenizer_config.json"]
    
    for file_name in required_files:
        if not (model_path / file_name).exists():
            logger.error(f"❌ 缺少模型文件: {file_name}")
            return False
    
    logger.info("✅ 模型文件完整")
    return True

def check_data():
    """检查数据文件"""
    logger.info("检查数据文件...")
    
    data_files = ["Datasets/train.csv", "Datasets/validation.csv", "Datasets/test.csv"]
    
    for file_path in data_files:
        if not Path(file_path).exists():
            logger.error(f"❌ 数据文件不存在: {file_path}")
            return False
    
    logger.info("✅ 数据文件存在")
    return True

def check_configs():
    """检查配置文件"""
    logger.info("检查配置文件...")
    
    config_files = [
        "configs/default_config.yaml",
        "configs/training_config.yaml", 
        "configs/evaluation_config.yaml"
    ]
    
    for file_path in config_files:
        if not Path(file_path).exists():
            logger.error(f"❌ 配置文件不存在: {file_path}")
            return False
    
    logger.info("✅ 配置文件存在")
    return True

def main():
    """主检查函数"""
    logger.info("="*50)
    logger.info("分子生成系统状态检查")
    logger.info("="*50)
    
    checks = [
        ("依赖包", check_dependencies),
        ("模型文件", check_model),
        ("数据文件", check_data),
        ("配置文件", check_configs)
    ]
    
    results = []
    for name, check_func in checks:
        logger.info(f"\n检查 {name}:")
        result = check_func()
        results.append(result)
    
    logger.info("\n" + "="*50)
    if all(results):
        logger.info("🎉 系统检查通过! 可以开始使用!")
        logger.info("\n推荐的下一步:")
        logger.info("  python train.py --config configs/training_config.yaml --debug")
        return True
    else:
        logger.error("❌ 系统检查未通过，请修复上述问题")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)