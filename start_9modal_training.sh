#!/bin/bash
# 9种模态组合训练启动脚本

echo "==============================================================="
echo "🚀 9种模态组合训练系统"
echo "支持: (SMILES/Graph/Image) × (SMILES/Graph/Image) = 9种组合"
echo "==============================================================="
echo ""

# 激活环境
source /root/miniconda3/bin/activate text2Mol

# 显示系统状态
echo "📊 系统状态:"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader)"
echo "磁盘: $(df -h /root/autodl-tmp | tail -1 | awk '{print $4 " 可用"}')"
echo ""

# 时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 训练配置
echo "🎯 选择训练配置:"
echo "1) 🧪 快速测试 (500样本, 2轮, ~15分钟)"
echo "2) 📊 标准训练 (2000样本, 5轮, ~1小时)"
echo "3) 💪 完整训练 (5000样本, 10轮, ~3小时)"
echo "4) 🔥 生产训练 (全部数据, 20轮, ~8小时)"
echo ""

read -p "请选择 [1-4]: " choice

case $choice in
    1)
        echo "✅ 快速测试模式"
        SAMPLE_SIZE=500
        EPOCHS=2
        BATCH_SIZE=8
        GRAD_ACCUM=1
        MODE="test"
        EST_TIME="15分钟"
        ;;
    2)
        echo "✅ 标准训练模式"
        SAMPLE_SIZE=2000
        EPOCHS=5
        BATCH_SIZE=8
        GRAD_ACCUM=2
        MODE="standard"
        EST_TIME="1小时"
        ;;
    3)
        echo "✅ 完整训练模式"
        SAMPLE_SIZE=5000
        EPOCHS=10
        BATCH_SIZE=8
        GRAD_ACCUM=3
        MODE="full"
        EST_TIME="3小时"
        ;;
    4)
        echo "✅ 生产训练模式"
        SAMPLE_SIZE=0
        EPOCHS=20
        BATCH_SIZE=8
        GRAD_ACCUM=4
        MODE="production"
        EST_TIME="8小时"
        ;;
    *)
        echo "❌ 无效选择，使用默认配置"
        SAMPLE_SIZE=1000
        EPOCHS=3
        BATCH_SIZE=8
        GRAD_ACCUM=2
        MODE="default"
        EST_TIME="30分钟"
        ;;
esac

# 输出目录
OUTPUT_DIR="/root/autodl-tmp/text2Mol-outputs/9modal_${TIMESTAMP}_${MODE}"
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/checkpoints"

echo ""
echo "📋 训练配置:"
echo "  模式: $MODE"
echo "  样本数: $([ $SAMPLE_SIZE -eq 0 ] && echo '全部(26K+)' || echo $SAMPLE_SIZE)"
echo "  训练轮数: $EPOCHS"
echo "  批大小: $BATCH_SIZE"
echo "  梯度累积: $GRAD_ACCUM"
echo "  有效批大小: $((BATCH_SIZE * GRAD_ACCUM))"
echo "  预计时间: $EST_TIME"
echo "  输出目录: $OUTPUT_DIR"
echo ""

# 构建训练命令
TRAIN_CMD="python train_9modal_fixed.py \
    --train-data Datasets/train.csv \
    --val-data Datasets/validation.csv \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation $GRAD_ACCUM \
    --epochs $EPOCHS \
    --lr 5e-5 \
    --mixed-precision \
    --num-workers 4 \
    --output-dir $OUTPUT_DIR \
    --save-interval 1 \
    --smiles-weight 1.0 \
    --graph-weight 0.7 \
    --image-weight 0.5"

if [ $SAMPLE_SIZE -gt 0 ]; then
    TRAIN_CMD="$TRAIN_CMD --sample-size $SAMPLE_SIZE"
fi

# 日志文件
LOG_FILE="$OUTPUT_DIR/logs/training.log"

echo "🚀 启动9模态训练..."
echo ""

# 启动训练
nohup $TRAIN_CMD > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!
echo $TRAIN_PID > "$OUTPUT_DIR/train.pid"

echo "✅ 训练已启动 (PID: $TRAIN_PID)"
echo ""

# 启动GPU监控
nohup bash -c "while kill -0 $TRAIN_PID 2>/dev/null; do 
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu --format=csv >> '$OUTPUT_DIR/logs/gpu.log'
    sleep 30
done" > /dev/null 2>&1 &
GPU_PID=$!
echo $GPU_PID > "$OUTPUT_DIR/gpu_monitor.pid"

echo "📊 GPU监控已启动 (PID: $GPU_PID)"
echo ""

# 创建状态检查脚本
cat > "$OUTPUT_DIR/check_status.sh" << 'EOF'
#!/bin/bash
# 9模态训练状态检查

clear
echo "==============================================================="
echo "📊 9种模态组合训练状态"
echo "==============================================================="
echo ""

DIR=$(dirname "$0")
TRAIN_PID=$(cat "$DIR/train.pid" 2>/dev/null)

# 检查进程
if [ -n "$TRAIN_PID" ] && kill -0 $TRAIN_PID 2>/dev/null; then
    echo "✅ 训练运行中 (PID: $TRAIN_PID)"
    RUNNING=true
else
    echo "⚠️ 训练已结束"
    RUNNING=false
fi

echo ""
echo "🔥 GPU状态:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader | \
    awk -F', ' '{printf "  使用率: %s | 内存: %s/%s | 温度: %s\n", $1, $2, $3, $4}'

echo ""
echo "💾 磁盘使用:"
du -sh "$DIR" 2>/dev/null | awk '{printf "  输出目录: %s\n", $1}'
df -h /root/autodl-tmp | tail -1 | awk '{printf "  剩余空间: %s\n", $4}'

if [ -f "$DIR/logs/training.log" ]; then
    echo ""
    echo "📈 训练进度:"
    
    # 提取最新epoch信息
    EPOCH_INFO=$(grep -o "Epoch [0-9]*/[0-9]*" "$DIR/logs/training.log" 2>/dev/null | tail -1)
    if [ -n "$EPOCH_INFO" ]; then
        echo "  当前: $EPOCH_INFO"
    fi
    
    # 提取损失信息
    LOSS_INFO=$(grep -E "Train Loss:|Val Loss:" "$DIR/logs/training.log" 2>/dev/null | tail -2)
    if [ -n "$LOSS_INFO" ]; then
        echo "$LOSS_INFO" | sed 's/^/  /'
    fi
    
    echo ""
    echo "📝 最新日志 (最后5行):"
    tail -5 "$DIR/logs/training.log" | sed 's/^/  /'
fi

# 检查最新的checkpoint
if [ -d "$DIR/checkpoints" ]; then
    LATEST_CKPT=$(ls -t "$DIR"/checkpoint_*.pth 2>/dev/null | head -1)
    if [ -n "$LATEST_CKPT" ]; then
        echo ""
        echo "💾 最新检查点:"
        echo "  $(basename $LATEST_CKPT)"
    fi
fi

echo ""
echo "==============================================================="

if [ "$RUNNING" = true ]; then
    echo "提示: 使用 'tail -f $DIR/logs/training.log' 查看实时日志"
else
    echo "训练已完成！查看 $DIR 获取结果"
fi
echo "==============================================================="
EOF

chmod +x "$OUTPUT_DIR/check_status.sh"

# 创建停止脚本
cat > "$OUTPUT_DIR/stop_training.sh" << 'EOF'
#!/bin/bash
DIR=$(dirname "$0")
TRAIN_PID=$(cat "$DIR/train.pid" 2>/dev/null)
GPU_PID=$(cat "$DIR/gpu_monitor.pid" 2>/dev/null)

echo "🛑 停止训练..."
[ -n "$TRAIN_PID" ] && kill $TRAIN_PID 2>/dev/null && echo "  训练进程已停止"
[ -n "$GPU_PID" ] && kill $GPU_PID 2>/dev/null && echo "  GPU监控已停止"
echo "✅ 完成"
EOF

chmod +x "$OUTPUT_DIR/stop_training.sh"

# 创建实时监控脚本
cat > "$OUTPUT_DIR/monitor.sh" << 'EOF'
#!/bin/bash
watch -n 5 "bash $(dirname $0)/check_status.sh"
EOF

chmod +x "$OUTPUT_DIR/monitor.sh"

echo "==============================================================="
echo "✅ 9种模态组合训练已启动!"
echo "==============================================================="
echo ""
echo "📊 支持的9种组合:"
echo "  输入: SMILES/Graph/Image (3种)"
echo "  输出: SMILES/Graph/Image (3种)"
echo "  总计: 3 × 3 = 9种组合"
echo ""
echo "🔧 管理命令:"
echo "  查看状态: $OUTPUT_DIR/check_status.sh"
echo "  实时监控: $OUTPUT_DIR/monitor.sh"
echo "  停止训练: $OUTPUT_DIR/stop_training.sh"
echo ""
echo "📝 日志查看:"
echo "  训练日志: tail -f $LOG_FILE"
echo "  GPU日志: tail -f $OUTPUT_DIR/logs/gpu.log"
echo ""
echo "💡 提示:"
echo "  - 训练在后台运行，可以安全关闭终端"
echo "  - 使用monitor.sh实时查看训练状态"
echo "  - 预计完成时间: $EST_TIME"
echo "==============================================================="