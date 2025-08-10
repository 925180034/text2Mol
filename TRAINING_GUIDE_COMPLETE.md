# 📚 9模态分子生成系统 - 完整训练指南

## 📋 目录
1. [核心训练文件](#核心训练文件)
2. [模型架构详解](#模型架构详解)
3. [训练参数说明](#训练参数说明)
4. [数据集配置](#数据集配置)
5. [训练启动方法](#训练启动方法)
6. [训练监控](#训练监控)
7. [模型保存与加载](#模型保存与加载)
8. [常见问题解决](#常见问题解决)

---

## 🗂️ 核心训练文件

### 1. 主要训练脚本

#### `train_9modal_fixed.py` (主训练脚本)
**功能**: 完整的9模态训练系统，支持所有输入输出组合
```python
# 文件位置: /root/text2Mol/scaffold-mol-generation/train_9modal_fixed.py
# 大小: ~20KB
# 作者: 最新修复版本 (2025-08-10)

# 核心类:
class NineModalityTrainer:
    - 支持9种输入输出组合
    - 混合精度训练
    - 自动保存检查点
    - TensorBoard日志
```

**使用方法**:
```bash
python train_9modal_fixed.py \
    --train-data Datasets/train.csv \
    --val-data Datasets/validation.csv \
    --batch-size 4 \
    --epochs 20 \
    --lr 5e-5 \
    --sample-size 10000 \
    --output-dir /root/autodl-tmp/text2Mol-outputs/
```

#### `train_fixed_multimodal.py` (单模态训练)
**功能**: 固定单一模态的训练，用于调试和优化
```python
# 特点:
- Token约束防止CUDA错误
- 简化的训练流程
- 适合初始实验
```

**使用方法**:
```bash
python train_fixed_multimodal.py \
    --scaffold-modality smiles \
    --batch-size 8 \
    --epochs 10
```

#### `train_joint_multimodal.py` (联合训练)
**功能**: 多模态联合训练，包含对齐损失
```python
# 特点:
- 跨模态对齐损失
- 联合优化所有模态
- 更好的特征学习
```

**使用方法**:
```bash
python train_joint_multimodal.py \
    --alignment-weight 0.1 \
    --batch-size 4 \
    --epochs 15
```

### 2. 启动脚本

#### `start_real_training.sh`
```bash
#!/bin/bash
# 交互式训练启动脚本
# 提供4种训练配置选项:
# 1) 快速测试 (100样本, 2轮)
# 2) 标准训练 (1000样本, 5轮)  
# 3) 生产训练 (5000样本, 10轮)
# 4) 完整训练 (全部数据, 20轮)

# 自动创建输出目录
# 生成监控脚本
# 保存训练配置
```

---

## 🏗️ 模型架构详解

### 整体架构
```
输入层 (3种模态):
├── SMILES Encoder (MolT5-base)
├── Graph Encoder (5层GIN)
└── Image Encoder (Swin Transformer)
      ↓
文本编码器:
└── Text Encoder (BERT/SciBERT)
      ↓
融合层:
└── Cross-Modal Fusion (Attention + Gating)
      ↓
适配层:
└── MolT5 Adapter (768→1024维度转换)
      ↓
生成器:
└── MolT5 Generator (Conditional Generation)
      ↓
输出解码器:
├── SMILES Decoder (直接输出)
├── Graph Decoder (开发中)
└── Image Decoder (开发中)
```

### 关键模型文件

#### 编码器 (`scaffold_mol_gen/models/encoders/`)
- `multimodal_encoder.py`: 多模态编码器管理器
- `smiles_encoder.py`: SMILES编码器 (MolT5)
- `graph_encoder.py`: 图编码器 (GIN)
- `image_encoder.py`: 图像编码器 (Swin)
- `text_encoder.py`: 文本编码器 (BERT)

#### 核心模型 (`scaffold_mol_gen/models/`)
- `end2end_model.py`: 端到端模型整合
- `fusion_simplified.py`: 模态融合层
- `molt5_adapter.py`: MolT5适配器
- `output_decoders.py`: 输出解码器

### 模型参数统计
```python
总参数: 596.52M
├── 可训练参数: 59.08M
└── 冻结参数: 537.44M

内存占用:
├── 模型: ~3GB
├── 批处理(batch_size=4): ~8GB
└── 梯度: ~2GB
```

---

## ⚙️ 训练参数说明

### 基础参数
| 参数 | 默认值 | 说明 | 推荐范围 |
|-----|-------|------|---------|
| `--batch-size` | 4 | 批大小 | 2-8 (取决于GPU内存) |
| `--epochs` | 10 | 训练轮数 | 5-20 |
| `--lr` | 5e-5 | 学习率 | 1e-5 到 1e-4 |
| `--gradient-accumulation` | 1 | 梯度累积步数 | 1-4 |
| `--warmup-steps` | 500 | 预热步数 | 100-1000 |
| `--weight-decay` | 1e-5 | 权重衰减 | 1e-6 到 1e-4 |

### 模型参数
| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--hidden-size` | 768 | 隐藏层维度 |
| `--num-heads` | 8 | 注意力头数 |
| `--num-layers` | 2 | Transformer层数 |
| `--dropout` | 0.1 | Dropout率 |
| `--max-seq-length` | 128 | 最大序列长度 |

### 训练策略参数
| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--mixed-precision` | True | 混合精度训练 |
| `--freeze-encoders` | True | 冻结预训练编码器 |
| `--freeze-molt5` | True | 冻结MolT5主体 |
| `--save-interval` | 1 | 保存间隔(轮) |
| `--sample-size` | 0 | 训练样本数(0=全部) |

### 损失权重
```python
loss_weights = {
    'smiles': 1.0,  # SMILES生成损失权重
    'graph': 0.5,   # 图生成损失权重
    'image': 0.3,   # 图像生成损失权重
    'alignment': 0.1 # 跨模态对齐损失权重
}
```

---

## 📊 数据集配置

### ChEBI-20数据集
```
位置: Datasets/
文件:
├── train.csv (21,487条, 25MB)
├── validation.csv (5,371条, 6.3MB)
└── test.csv (3,297条, 3.9MB)

字段说明:
- CID: 化合物ID
- SMILES: 完整分子SMILES
- description: 分子描述文本
- scaffold: Murcko scaffold (运行时提取)
```

### 数据预处理
```python
# 自动处理:
1. SMILES验证和规范化
2. Murcko scaffold提取
3. 文本tokenization
4. 图结构生成(动态)
5. 图像生成(动态)
```

---

## 🚀 训练启动方法

### 方法1: 使用启动脚本 (推荐)
```bash
# 交互式选择配置
./start_real_training.sh

# 选择选项:
# 1) 快速测试 - 验证系统工作
# 2) 标准训练 - 日常实验
# 3) 生产训练 - 正式模型
# 4) 完整训练 - 最终模型
```

### 方法2: 直接运行Python脚本
```bash
# 基础训练
python train_9modal_fixed.py \
    --batch-size 4 \
    --epochs 10

# 带所有参数的训练
python train_9modal_fixed.py \
    --train-data Datasets/train.csv \
    --val-data Datasets/validation.csv \
    --batch-size 4 \
    --epochs 20 \
    --lr 5e-5 \
    --gradient-accumulation 2 \
    --warmup-steps 500 \
    --weight-decay 1e-5 \
    --mixed-precision \
    --save-interval 1 \
    --output-dir /root/autodl-tmp/text2Mol-outputs/my_experiment/
```

### 方法3: 后台训练
```bash
# 使用nohup后台运行
nohup python train_9modal_fixed.py \
    --batch-size 4 \
    --epochs 20 \
    > training.log 2>&1 &

# 查看进程
ps aux | grep train_9modal
```

---

## 📈 训练监控

### TensorBoard监控
```bash
# 启动TensorBoard
tensorboard --logdir /root/autodl-tmp/text2Mol-outputs/*/tensorboard

# 访问: http://localhost:6006
```

### 日志文件
```bash
# 实时查看训练日志
tail -f /root/autodl-tmp/text2Mol-outputs/*/logs/training.log

# 查看损失变化
grep "Loss" training.log | tail -20
```

### 监控脚本
```bash
# 运行生成的监控脚本
/root/autodl-tmp/text2Mol-outputs/*/monitor.sh
```

### GPU监控
```bash
# 实时GPU使用情况
watch -n 2 nvidia-smi

# 或使用
nvidia-smi -l 2
```

---

## 💾 模型保存与加载

### 保存位置
```
/root/autodl-tmp/text2Mol-outputs/{timestamp}/
├── checkpoints/
│   ├── epoch_1.pth
│   ├── epoch_2.pth
│   └── best_model.pth
├── logs/
│   └── training.log
├── tensorboard/
│   └── events.out.tfevents.*
└── training_config.txt
```

### 检查点格式
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_loss,
    'val_loss': val_loss,
    'best_val_loss': best_val_loss,
    'config': config
}
```

### 加载模型
```python
# 加载训练好的模型
import torch
from scaffold_mol_gen.models.end2end_model import End2EndMolecularGenerator

# 创建模型
model = End2EndMolecularGenerator(
    hidden_size=768,
    molt5_path="/root/autodl-tmp/text2Mol-models/molt5-base",
    device='cuda'
)

# 加载权重
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

---

## 🔧 常见问题解决

### 1. CUDA内存不足
```bash
# 解决方案:
- 减小batch_size (如: 4→2)
- 启用梯度累积
- 使用混合精度训练
- 清理GPU缓存: torch.cuda.empty_cache()
```

### 2. 生成质量差
```bash
# 解决方案:
- 增加训练轮数 (至少10-20轮)
- 使用更多训练数据
- 调整学习率 (尝试1e-5)
- 解冻部分预训练参数
```

### 3. 训练不收敛
```bash
# 解决方案:
- 检查数据质量
- 降低学习率
- 增加warmup步数
- 使用梯度裁剪
```

### 4. 训练速度慢
```bash
# 解决方案:
- 启用混合精度训练
- 增大batch_size
- 使用梯度累积
- 减少验证频率
```

---

## 📝 推荐训练流程

### 第1阶段: 验证系统
```bash
# 100个样本，2轮，验证系统工作
python train_9modal_fixed.py \
    --sample-size 100 \
    --epochs 2 \
    --batch-size 4
```

### 第2阶段: 初步训练
```bash
# 1000个样本，5轮，初步结果
python train_9modal_fixed.py \
    --sample-size 1000 \
    --epochs 5 \
    --batch-size 4
```

### 第3阶段: 正式训练
```bash
# 10000个样本，20轮，正式模型
python train_9modal_fixed.py \
    --sample-size 10000 \
    --epochs 20 \
    --batch-size 4 \
    --lr 1e-5
```

### 第4阶段: 微调优化
```bash
# 解冻部分参数，继续训练
python train_9modal_fixed.py \
    --resume-from best_model.pth \
    --freeze-encoders False \
    --lr 5e-6 \
    --epochs 5
```

---

## 📊 预期结果

### 训练损失曲线
```
Epoch 1: Loss ~100 → ~60
Epoch 5: Loss ~60 → ~20
Epoch 10: Loss ~20 → ~5
Epoch 20: Loss ~5 → ~2
```

### 生成质量指标
```
有效性(Validity): >90%
唯一性(Uniqueness): >95%
Scaffold保持率: >80%
相似度(Tanimoto): >0.6
```

---

## 🎯 最佳实践

1. **开始时冻结预训练模型**，后期微调
2. **使用梯度累积**模拟更大batch size
3. **保存所有检查点**，不只是最佳模型
4. **监控验证损失**防止过拟合
5. **定期评估生成质量**，不只看损失

---

## 📧 技术支持

如有问题，请检查:
1. CLAUDE.md - 项目特定指南
2. docs/reports/ - 详细技术报告
3. 训练日志文件 - 具体错误信息

---

*最后更新: 2025-08-10*
*版本: v1.0*