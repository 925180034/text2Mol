# 后台训练完整指南

## 🚀 快速开始

### 一键启动后台训练+监控
```bash
# 终端1: 启动闪电验证训练（45分钟）
python background_training.py smiles

# 终端2: 启动实时监控
python training_monitor.py
```

## 📋 完整功能

### 🔥 后台训练启动器 (`background_training.py`)

**交互模式**:
```bash
python background_training.py
```
选项：
1. **闪电验证** - SMILES模态，45分钟
2. **完整训练** - 三模态顺序，3小时
3. **自定义训练** - 选择特定模态
4. **查看状态** - 当前训练状态
5. **停止训练** - 停止所有任务

**命令行模式**:
```bash
# 启动特定模态
python background_training.py smiles
python background_training.py graph  
python background_training.py image

# 查看状态
python background_training.py --status

# 停止训练
python background_training.py --stop          # 停止所有
python background_training.py --stop smiles   # 停止特定模态
```

### 📊 实时监控器 (`training_monitor.py`)

**功能特性**:
- 🔄 实时训练进度（epoch、loss、有效性）
- 🖥️ GPU使用率和显存监控
- 💾 磁盘空间监控
- 📝 最新日志显示
- ⚡ 进程状态监控

**启动监控**:
```bash
python training_monitor.py
# 或
python background_training.py --monitor
```

## 🎯 推荐使用流程

### 方案1：首次验证（推荐）
```bash
# Step 1: 启动闪电验证（45分钟）
python background_training.py smiles

# Step 2: 新终端启动监控
python training_monitor.py

# Step 3: 等待完成，评估效果
python final_fixed_evaluation.py --num_samples 50
```

### 方案2：完整训练
```bash
# Step 1: 启动完整后台训练
python background_training.py
# 选择选项2（完整训练）

# Step 2: 实时监控
python training_monitor.py

# Step 3: 多模态评估
python demo_multimodal_evaluation.py
```

## 📊 监控界面预览

```
🚀 多模态训练实时监控
================================================================================
时间: 2025-08-05 17:30:15

📊 训练任务状态:
----------------------------------------
  🔄 SMILES: 运行中 (PID: 12345)
     轮次: 2/5
     损失: 0.3245
     有效性: 82.3%
     CPU: 15.2%
     内存: 1024MB

🖥️  GPU状态:
----------------------------------------
  GPU 0: NVIDIA vGPU-32GB
     利用率: 85%
     显存: 8432MB / 32760MB
     温度: 45°C

💾 磁盘状态:
----------------------------------------
  数据盘: 22G / 50G (44%)

📝 最近日志:
----------------------------------------
  SMILES:
    Epoch 2/5 - Step 1200/2400 - Loss: 0.3245
    Validation - Validity: 82.3% - Uniqueness: 91.2%
    
================================================================================
💡 命令:
  Ctrl+C: 退出监控
  python background_training.py --status: 查看详细状态
  python background_training.py --stop: 停止所有训练
```

## 🎛️ 训练控制

### 查看状态
```bash
python background_training.py --status
```

### 停止训练
```bash
# 停止所有训练
python background_training.py --stop

# 停止特定模态
python background_training.py --stop smiles
python background_training.py --stop graph
python background_training.py --stop image
```

### 查看日志
```bash
# 实时查看最新日志
tail -f logs/bg_smiles_*.log
tail -f logs/bg_graph_*.log
tail -f logs/bg_image_*.log

# 查看错误信息
grep -i error logs/bg_*.log
```

## 📁 文件组织

### 日志文件
```
logs/
├── bg_smiles_20250805_173015.log    # SMILES训练日志
├── bg_graph_20250805_174520.log     # Graph训练日志  
├── bg_image_20250805_180130.log     # Image训练日志
└── training_pids.json               # 进程ID信息
```

### 输出文件
```
/root/autodl-tmp/text2Mol-outputs/
├── bg_smiles/                       # SMILES模态输出
├── bg_graph/                        # Graph模态输出
└── bg_image/                        # Image模态输出
```

## ⚡ 性能优化

### 32GB显卡优化
- **SMILES**: batch_size=20 (vs 8)
- **Graph**: batch_size=12 (vs 4)
- **Image**: batch_size=8 (vs 2)
- **学习率**: 1e-4 (vs 5e-5)
- **混合精度**: 启用FP16

### 预期时间
| 模态 | 后台训练时间 | 加速比例 |
|------|-------------|----------|
| SMILES | 30-45分钟 | 4-6倍 |
| Graph | 1-1.5小时 | 4倍 |
| Image | 1.5-2小时 | 4倍 |
| **总计** | **2-3小时** | **4倍** |

## 🔧 故障排除

### 常见问题

**1. 进程意外停止**
```bash
# 检查是否还在运行
python background_training.py --status

# 查看错误日志
tail -100 logs/bg_*.log
```

**2. GPU内存不足**
```bash
# 检查GPU状态
nvidia-smi

# 清理GPU内存
python -c "import torch; torch.cuda.empty_cache()"
```

**3. 磁盘空间不足**
```bash
# 检查磁盘使用
df -h /root/autodl-tmp

# 清理旧文件
rm -rf /root/autodl-tmp/text2Mol-outputs/old_*
```

### 紧急停止
```bash
# 强制停止所有训练
pkill -f train_multimodal.py

# 清理PID文件
rm -f logs/training_pids.json
```

## 🎉 训练完成后

### 立即可用的评估
```bash
# 快速评估单模态效果
python final_fixed_evaluation.py --num_samples 50

# 多模态对比评估
python demo_multimodal_evaluation.py

# 完整评估报告
python run_multimodal_evaluation.py
```

### 模型使用
训练完成的模型保存在:
- `/root/autodl-tmp/text2Mol-outputs/bg_smiles/`
- `/root/autodl-tmp/text2Mol-outputs/bg_graph/`
- `/root/autodl-tmp/text2Mol-outputs/bg_image/`

现在就可以开始后台训练了！建议先用闪电验证模式试试效果。