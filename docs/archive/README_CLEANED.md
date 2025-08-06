# Text2Mol 多模态分子生成系统 - 清理后结构

## 🎯 核心运行脚本

### 主要训练系统
- **`train_multimodal.py`** - 主训练脚本，支持SMILES/Graph/Image三种模态
- **`safe_background_training.py`** - 安全后台训练启动器，支持双GPU并行
- **`training_monitor.py`** - 实时训练状态监控
- **`auto_cleanup_daemon.py`** - 自动磁盘空间清理守护进程

## 🧪 测试和评估脚本

### 验证和评估工具
- **`test_all_modalities.py`** - 全模态测试脚本，验证三种输入模态
- **`multimodal_evaluation.py`** - 完整多模态评估，支持7种输入输出组合
- **`test_trained_multimodal.py`** - 训练模型测试脚本
- **`quick_test_training.py`** - 快速训练测试工具

## 📁 项目目录结构

```
scaffold-mol-generation/
├── scaffold_mol_gen/          # 核心代码包
│   ├── models/                # 神经网络模型
│   ├── data/                  # 数据处理
│   ├── training/              # 训练系统
│   └── utils/                 # 工具函数
├── tools/                     # 辅助工具脚本 (21个)
├── archive/                   # 归档文件
│   └── test_scripts/          # 旧测试脚本存档
├── logs/                      # 训练日志
├── Datasets/                  # ChEBI-20数据集
└── configs/                   # 配置文件

核心脚本: 8个 (清理率76%)
```

## 🚀 使用指南

### 启动训练
```bash
# 单模态训练
python safe_background_training.py smiles
python safe_background_training.py graph  
python safe_background_training.py image

# 监控训练状态
python training_monitor.py
```

### 测试和评估
```bash
# 测试所有模态
python test_all_modalities.py

# 完整评估
python multimodal_evaluation.py

# 快速测试
python quick_test_training.py
```

## ✅ 清理成果
- **文件减少**: 33个 → 8个 (76%减少)
- **结构优化**: 核心脚本与工具脚本分离
- **空间释放**: 显著减少磁盘占用
- **维护性提升**: 清晰的文件组织结构

---
*清理完成时间: 2025-08-06*
*当前训练状态: Image模态训练进行中*