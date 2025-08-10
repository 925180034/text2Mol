# 🔍 训练监控完整指南

## 一、快速监控命令

### 1️⃣ 检查是否有训练在运行
```bash
ps aux | grep -E "python.*train" | grep -v grep
```

### 2️⃣ 查看GPU使用情况
```bash
# 单次查看
nvidia-smi

# 实时监控（每2秒刷新）
watch -n 2 nvidia-smi

# 简洁版GPU监控
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits
```

### 3️⃣ 查看最新的训练日志
```bash
# 找到最新的训练目录
ls -lt /root/autodl-tmp/text2Mol-outputs/ | grep -E "modal|training" | head -1

# 查看最新日志（替换目录名）
tail -f /root/autodl-tmp/text2Mol-outputs/9modal_*/logs/training.log

# 只看错误和警告
tail -f /root/autodl-tmp/text2Mol-outputs/9modal_*/logs/training.log | grep -E "ERROR|WARNING"

# 只看进度
tail -f /root/autodl-tmp/text2Mol-outputs/9modal_*/logs/training.log | grep -E "Epoch|Loss"
```

### 4️⃣ 系统资源监控
```bash
# CPU和内存
htop

# 磁盘使用
df -h /root/autodl-tmp

# 查看训练输出目录大小
du -sh /root/autodl-tmp/text2Mol-outputs/9modal_*
```

## 二、使用自动监控脚本

### 🔧 完整系统监控
```bash
./monitor_all.sh
```
显示：
- 所有Python训练进程
- GPU状态
- 最新训练目录和状态
- 磁盘使用
- 系统负载

### 🚀 一键启动并监控
```bash
./run_and_monitor.sh
```
功能：
- 自动启动训练
- 实时监控面板
- 每10秒自动刷新
- 显示训练进度和损失

### 📊 使用训练脚本的监控
```bash
# 启动训练
./start_9modal_training.sh

# 然后使用生成的监控脚本
/root/autodl-tmp/text2Mol-outputs/9modal_*/check_status.sh
/root/autodl-tmp/text2Mol-outputs/9modal_*/monitor.sh
```

## 三、高级监控技巧

### 📈 TensorBoard监控
```bash
# 启动TensorBoard
tensorboard --logdir /root/autodl-tmp/text2Mol-outputs/9modal_*/tensorboard --port 6006

# 然后在浏览器访问 http://localhost:6006
```

### 🔄 持续监控命令组合
```bash
# 在tmux中打开多个窗口监控
tmux new-session -s monitor

# 窗口1: GPU监控
watch -n 2 nvidia-smi

# 窗口2: 日志监控
tail -f /root/autodl-tmp/text2Mol-outputs/9modal_*/logs/training.log

# 窗口3: 系统监控
htop
```

### 📝 提取训练统计
```bash
# 查看所有epoch的损失
grep "Epoch.*Loss" /root/autodl-tmp/text2Mol-outputs/9modal_*/logs/training.log

# 查看最佳验证损失
grep "best" /root/autodl-tmp/text2Mol-outputs/9modal_*/logs/training.log

# 统计训练时间
head -1 /root/autodl-tmp/text2Mol-outputs/9modal_*/logs/training.log
tail -1 /root/autodl-tmp/text2Mol-outputs/9modal_*/logs/training.log
```

## 四、问题排查

### ❌ 训练没有启动
```bash
# 1. 检查进程
ps aux | grep python

# 2. 查看错误日志
tail -100 /root/autodl-tmp/text2Mol-outputs/9modal_*/logs/training.log | grep -E "ERROR|Error|Traceback"

# 3. 检查GPU是否可用
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### ⚠️ 训练中断
```bash
# 查看最后的日志
tail -50 /root/autodl-tmp/text2Mol-outputs/9modal_*/logs/training.log

# 检查系统日志
dmesg | tail -20

# 检查内存
free -h
```

### 🔍 查找特定训练
```bash
# 按时间查找
find /root/autodl-tmp/text2Mol-outputs -name "*9modal*" -type d -mtime -1

# 查找运行中的训练
for dir in /root/autodl-tmp/text2Mol-outputs/*/; do
    if [ -f "$dir/train.pid" ]; then
        PID=$(cat "$dir/train.pid")
        if kill -0 $PID 2>/dev/null; then
            echo "运行中: $dir (PID: $PID)"
        fi
    fi
done
```

## 五、常用组合命令

### 🎯 完整状态检查（一行命令）
```bash
echo "=== GPU ===" && nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader && echo "=== 进程 ===" && ps aux | grep "train.*py" | grep -v grep && echo "=== 最新日志 ===" && tail -3 /root/autodl-tmp/text2Mol-outputs/9modal_*/logs/training.log 2>/dev/null
```

### 📊 训练进度追踪
```bash
watch -n 10 'grep "Epoch" /root/autodl-tmp/text2Mol-outputs/9modal_*/logs/training.log | tail -1'
```

### 💾 资源使用监控
```bash
watch -n 5 'nvidia-smi --query-gpu=memory.used --format=csv,noheader && df -h /root/autodl-tmp | tail -1'
```

---

## 🚀 最简单的方法

如果你只想快速知道训练状态，运行：

```bash
./monitor_all.sh
```

这会显示所有你需要的信息！