# 🚀 多模态分子生成模型训练指南

## 📋 前置准备

### 1. 环境检查
```bash
# 检查GPU
nvidia-smi

# 检查数据文件
ls -lh Datasets/*.csv

# 检查预训练模型
ls -lh /root/autodl-tmp/text2Mol-models/
```

## 🎯 训练方案选择

### 方案1：快速启动（推荐）
最简单的方式，一键启动后台训练：
```bash
./start_background_training.sh
```

### 方案2：交互式启动
提供多种训练模式选择：
```bash
./launch_production_training.sh
```
然后选择：
- 1 = 固定单模态训练（稳定）
- 2 = 联合多模态训练（推荐）
- 3 = 两阶段训练（最优）

### 方案3：自定义启动
完全控制训练参数：

#### 单模态训练（更稳定）
```bash
nohup python train_fixed_multimodal.py \
    --batch-size 12 \
    --gradient-accumulation 2 \
    --epochs 20 \
    --lr 5e-5 \
    --output-dir /root/autodl-tmp/text2Mol-outputs/my_training \
    > training.log 2>&1 &
```

#### 联合多模态训练（更强大）
```bash
nohup python train_joint_multimodal.py \
    --batch-size 16 \
    --gradient-accumulation 2 \
    --epochs 20 \
    --lr 5e-5 \
    --alignment-weight 0.1 \
    --output-dir /root/autodl-tmp/text2Mol-outputs/joint_training \
    > training.log 2>&1 &
```

## 📊 监控训练

### 实时监控
```bash
# 使用Python监控脚本（推荐）
python monitor_training.py

# 查看所有训练
python monitor_training.py --list

# 监控指定目录
python monitor_training.py --dir /root/autodl-tmp/text2Mol-outputs/production_20250808_120000
```

### 日志查看
```bash
# 查看最新日志
tail -f /root/autodl-tmp/text2Mol-outputs/*/training.log

# 查看GPU使用
watch -n 1 nvidia-smi
```

### TensorBoard
```bash
# 启动TensorBoard
tensorboard --logdir /root/autodl-tmp/text2Mol-outputs/*/tensorboard --port 6006

# 然后在浏览器访问 http://localhost:6006
```

## 🔧 参数优化建议

### 32GB GPU最优配置
```python
config = {
    'batch_size': 16,               # GPU内存允许的最大批次
    'gradient_accumulation': 2,     # 有效批次 = 16 * 2 = 32
    'learning_rate': 5e-5,          # 标准学习率
    'warmup_steps': 1000,           # 预热步数
    'epochs': 20,                   # 完整训练
    'alignment_weight': 0.1,        # 模态对齐权重
    'num_workers': 4,               # 数据加载并行
}
```

### 内存不足调整
如果遇到OOM错误：
1. 减小batch_size（12 → 8 → 4）
2. 增加gradient_accumulation
3. 减少max_text_length（128 → 96）
4. 减少max_smiles_length（128 → 96）

## 🎓 训练模式详解

### 单模态训练
- **优点**：稳定，收敛快，内存占用少
- **缺点**：不能利用多模态互补信息
- **适用**：初次训练，验证系统

### 联合多模态训练
- **优点**：充分利用所有模态，性能最优
- **缺点**：训练时间长，需要更多内存
- **适用**：生产环境，追求最佳性能

### 两阶段训练
- **第一阶段**：单模态预热（5 epochs）
- **第二阶段**：多模态微调（15 epochs）
- **优点**：结合两者优势
- **适用**：时间充足，追求最优结果

## 📈 预期性能

### 训练时间估算（32GB GPU）
- 单模态：~2小时/epoch（全数据集）
- 多模态：~3小时/epoch（全数据集）
- 完整训练（20 epochs）：40-60小时

### 预期指标
- Validity: 0.85-0.95
- Uniqueness: 0.80-0.90
- BLEU: 0.70-0.80
- Fingerprint Similarity: 0.75-0.85

## 🚨 常见问题

### 1. CUDA Out of Memory
```bash
# 解决方案
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# 减小batch_size
```

### 2. 训练中断恢复
```bash
# 从检查点恢复（功能开发中）
python train_joint_multimodal.py \
    --resume-from /path/to/checkpoint.pt
```

### 3. 查看训练进程
```bash
# 查看进程
ps aux | grep train

# 查看PID
cat /root/autodl-tmp/text2Mol-outputs/*/train.pid
```

### 4. 停止训练
```bash
# 优雅停止
kill $(cat /root/autodl-tmp/text2Mol-outputs/*/train.pid)

# 强制停止
kill -9 $(cat /root/autodl-tmp/text2Mol-outputs/*/train.pid)
```

## 💾 输出文件说明

训练会生成以下文件：
```
output_dir/
├── config.json           # 训练配置
├── training.log          # 训练日志
├── train.pid            # 进程ID
├── training_info.json   # 训练信息
├── training_status.json # 实时状态
├── tensorboard/         # TensorBoard日志
├── checkpoints/         # 模型检查点
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   └── ...
├── best_model.pt        # 最佳模型
└── final_model.pt       # 最终模型
```

## 🎯 下一步

训练完成后：
1. 评估模型性能：使用评估脚本测试
2. 生成示例：测试不同模态输入
3. 部署应用：集成到下游任务

## 📞 需要帮助？

- 查看日志文件了解详细错误
- 使用监控脚本实时查看状态
- 检查GPU内存使用情况
- 确保数据文件路径正确