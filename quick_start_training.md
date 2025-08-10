# 🚀 快速开始后台训练和监控

## 一、最简单的方法 - 使用一键启动脚本

```bash
# 进入项目目录
cd /root/text2Mol/scaffold-mol-generation

# 运行后台训练启动器
./start_background_training.sh
```

启动后选择训练模式：
- **1** - 快速测试（10分钟，验证系统）
- **2** - 标准训练（2小时，基础模型）
- **3** - 完整训练（8小时，生产模型）
- **4** - 高性能训练（大批次，最佳效果）

脚本会自动：
- ✅ 启动后台训练进程
- ✅ 创建监控系统
- ✅ 生成管理脚本
- ✅ 设置日志记录

---

## 二、手动启动后台训练

### 1. 快速测试训练（验证系统工作）

```bash
# 100个样本，2轮训练，约10分钟
nohup python train_9_modalities.py \
    --train-data Datasets/train.csv \
    --val-data Datasets/validation.csv \
    --batch-size 8 \
    --epochs 2 \
    --sample-size 100 \
    --output-dir /root/autodl-tmp/text2Mol-outputs/test_$(date +%Y%m%d_%H%M%S) \
    > training.log 2>&1 &

# 记录PID
echo $! > train.pid
```

### 2. 标准训练（推荐）

```bash
# 5000个样本，5轮训练，约2小时
nohup python train_9_modalities.py \
    --train-data Datasets/train.csv \
    --val-data Datasets/validation.csv \
    --batch-size 16 \
    --gradient-accumulation 2 \
    --epochs 5 \
    --sample-size 5000 \
    --lr 5e-5 \
    --mixed-precision \
    --output-dir /root/autodl-tmp/text2Mol-outputs/standard_$(date +%Y%m%d_%H%M%S) \
    > training.log 2>&1 &
```

### 3. 完整训练（生产级）

```bash
# 全部数据，10轮训练，约8小时
nohup python train_9_modalities.py \
    --train-data Datasets/train.csv \
    --val-data Datasets/validation.csv \
    --batch-size 16 \
    --gradient-accumulation 3 \
    --epochs 10 \
    --lr 5e-5 \
    --mixed-precision \
    --num-workers 4 \
    --output-dir /root/autodl-tmp/text2Mol-outputs/full_$(date +%Y%m%d_%H%M%S) \
    --save-steps 500 \
    --smiles-weight 1.0 \
    --graph-weight 0.7 \
    --image-weight 0.5 \
    > training.log 2>&1 &
```

---

## 三、监控训练进度

### 方法1：实时监控仪表板（推荐）

```bash
# 启动实时监控（每10秒更新）
python real_time_monitor.py --output-dir /root/autodl-tmp/text2Mol-outputs/你的训练目录 --interval 10
```

显示内容：
- 🔥 GPU使用率、温度、内存
- 💻 CPU、RAM、磁盘使用
- 📈 训练损失、验证损失
- ⚠️ 系统告警

### 方法2：查看训练日志

```bash
# 实时查看训练日志
tail -f training.log

# 只看损失信息
tail -f training.log | grep -E "Loss|Epoch"

# 查看验证结果
tail -f training.log | grep "Validation"
```

### 方法3：监控GPU使用

```bash
# 实时GPU监控
watch -n 5 nvidia-smi

# 或者更详细的信息
nvidia-smi -l 5
```

### 方法4：TensorBoard可视化

```bash
# 启动TensorBoard
tensorboard --logdir /root/autodl-tmp/text2Mol-outputs/你的训练目录/tensorboard --port 6006

# 然后在浏览器访问 http://localhost:6006
```

---

## 四、管理训练进程

### 查看训练状态

```bash
# 如果使用了启动脚本，直接运行
/root/autodl-tmp/text2Mol-outputs/你的训练目录/check_status.sh

# 或手动查看进程
ps aux | grep train_9_modalities
```

### 停止训练

```bash
# 如果使用了启动脚本
/root/autodl-tmp/text2Mol-outputs/你的训练目录/stop_training.sh

# 或手动停止
kill $(cat train.pid)
```

### 恢复训练（从检查点）

```bash
python train_9_modalities.py \
    --resume-from /root/autodl-tmp/text2Mol-outputs/你的训练目录/checkpoints/最新检查点.pth \
    [其他原始参数...]
```

---

## 五、常见问题

### 1. CUDA内存不足

减小批大小或增加梯度累积：
```bash
--batch-size 8 --gradient-accumulation 4  # 有效批大小32
```

### 2. 训练太慢

- 启用混合精度：`--mixed-precision`
- 增加数据加载进程：`--num-workers 8`
- 减少验证频率：`--val-interval 1000`

### 3. 查看训练是否正常

```bash
# 检查最新的训练日志
tail -n 50 training.log

# 检查GPU是否在使用
nvidia-smi

# 检查输出目录大小
du -sh /root/autodl-tmp/text2Mol-outputs/你的训练目录
```

---

## 六、推荐配置

### 快速验证（10分钟）
```bash
./start_background_training.sh
# 选择 1
```

### 日常开发（2小时）
```bash
./start_background_training.sh
# 选择 2
```

### 生产训练（8小时）
```bash
./start_background_training.sh
# 选择 3
```

---

## 🎯 现在就开始！

最简单的开始方式：
```bash
cd /root/text2Mol/scaffold-mol-generation
./start_background_training.sh
# 选择 1 进行快速测试
```

训练会在后台运行，你可以：
- 关闭终端，训练继续
- 随时查看进度
- 实时监控性能
- 安全停止/恢复