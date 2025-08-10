#!/bin/bash
# 真正的9模态训练启动脚本

echo "==============================================================="
echo "🚀 9模态分子生成系统 - 生产训练"
echo "==============================================================="
echo ""
echo "📊 训练配置："
echo "  - 9种输入输出组合：(SMILES/Graph/Image) × (SMILES/Graph/Image)"
echo "  - 数据集：ChEBI-20 (21487训练样本)"
echo "  - GPU：32GB NVIDIA vGPU"
echo ""

# 时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/root/autodl-tmp/text2Mol-outputs/9modal_${TIMESTAMP}_production"

# 创建目录
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/checkpoints"
mkdir -p "$OUTPUT_DIR/tensorboard"

echo "📁 输出目录: $OUTPUT_DIR"
echo ""

# 选择训练配置
echo "🎯 选择训练配置:"
echo ""
echo "1) 🧪 快速测试 (100样本, 2轮, ~10分钟)"
echo "2) 📊 标准训练 (1000样本, 5轮, ~30分钟)"  
echo "3) 💪 生产训练 (5000样本, 10轮, ~2小时)"
echo "4) 🔥 完整训练 (全部数据, 20轮, ~6小时)"
echo ""
read -p "请选择 [1-4]: " choice

case $choice in
    1)
        SAMPLE_SIZE=100
        EPOCHS=2
        BATCH_SIZE=4
        DESC="快速测试"
        ;;
    2)
        SAMPLE_SIZE=1000
        EPOCHS=5
        BATCH_SIZE=4
        DESC="标准训练"
        ;;
    3)
        SAMPLE_SIZE=5000
        EPOCHS=10
        BATCH_SIZE=4
        DESC="生产训练"
        ;;
    4)
        SAMPLE_SIZE=0  # 0表示全部数据
        EPOCHS=20
        BATCH_SIZE=4
        DESC="完整训练"
        ;;
    *)
        echo "无效选择，使用默认配置"
        SAMPLE_SIZE=1000
        EPOCHS=5
        BATCH_SIZE=4
        DESC="标准训练"
        ;;
esac

echo ""
echo "✅ 选择了: $DESC"
echo "  - 样本数: ${SAMPLE_SIZE:-全部}"
echo "  - 训练轮数: $EPOCHS"
echo "  - 批大小: $BATCH_SIZE"
echo ""

# 保存配置
cat > "$OUTPUT_DIR/training_config.txt" << EOF
训练配置: $DESC
样本数: ${SAMPLE_SIZE:-全部}
训练轮数: $EPOCHS
批大小: $BATCH_SIZE
开始时间: $(date)
输出目录: $OUTPUT_DIR
EOF

# 启动训练
echo "🔄 启动训练..."
nohup /root/miniconda3/envs/text2Mol/bin/python train_9modal_fixed.py \
    --train-data Datasets/train.csv \
    --val-data Datasets/validation.csv \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --sample-size $SAMPLE_SIZE \
    --lr 5e-5 \
    --gradient-accumulation 1 \
    --save-interval 1 \
    --output-dir "$OUTPUT_DIR" \
    > "$OUTPUT_DIR/logs/training.log" 2>&1 &

TRAIN_PID=$!
echo $TRAIN_PID > "$OUTPUT_DIR/train.pid"

echo "✅ 训练已启动 (PID: $TRAIN_PID)"
echo ""

# 创建监控脚本
cat > "$OUTPUT_DIR/monitor.sh" << 'EOF'
#!/bin/bash
# 监控训练进度

clear
echo "==============================================================="
echo "📊 9模态训练监控"
echo "==============================================================="
echo ""

# 检查进程
PID=$(cat train.pid 2>/dev/null)
if [ -n "$PID" ] && kill -0 $PID 2>/dev/null; then
    echo "✅ 训练运行中 (PID: $PID)"
else
    echo "🔴 训练已停止"
fi
echo ""

# GPU状态
echo "🔥 GPU状态:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo ""

# 最新日志
echo "📝 最新训练日志:"
tail -5 logs/training.log 2>/dev/null | grep -E "Epoch|Loss|INFO"
echo ""

# 磁盘状态
echo "💾 磁盘使用:"
du -sh . 2>/dev/null
echo ""

echo "==============================================================="
echo "操作："
echo "  查看完整日志: tail -f logs/training.log"
echo "  停止训练: kill $PID"
echo "==============================================================="
EOF

chmod +x "$OUTPUT_DIR/monitor.sh"

# 等待训练开始
echo "⏳ 等待训练初始化..."
sleep 5

# 显示初始状态
echo ""
echo "📊 初始状态："
echo "==============================================================="

# 检查进程
if kill -0 $TRAIN_PID 2>/dev/null; then
    echo "✅ 训练进程运行中"
    
    # 显示GPU使用
    echo ""
    echo "GPU使用:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
    
    # 显示最新日志
    echo ""
    echo "最新日志:"
    tail -10 "$OUTPUT_DIR/logs/training.log" 2>/dev/null | head -5
else
    echo "❌ 训练启动失败"
    echo "查看错误: tail -20 $OUTPUT_DIR/logs/training.log"
fi

echo ""
echo "==============================================================="
echo "📋 监控命令："
echo ""
echo "  1. 实时监控: $OUTPUT_DIR/monitor.sh"
echo "  2. 查看日志: tail -f $OUTPUT_DIR/logs/training.log"
echo "  3. GPU监控: watch -n 2 nvidia-smi"
echo "  4. TensorBoard: tensorboard --logdir $OUTPUT_DIR/tensorboard"
echo "  5. 停止训练: kill $TRAIN_PID"
echo ""
echo "==============================================================="