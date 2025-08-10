#!/bin/bash
# 超简单的9模态训练启动脚本

echo "🚀 启动9模态训练（快速测试版）"
echo "================================"

# 时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/root/autodl-tmp/text2Mol-outputs/9modal_${TIMESTAMP}_quick"

# 创建目录
mkdir -p "$OUTPUT_DIR/logs"

echo "📁 输出目录: $OUTPUT_DIR"
echo "🔄 启动训练..."

# 启动训练 - 使用最小配置快速测试
nohup /root/miniconda3/envs/text2Mol/bin/python train_9modal_fixed.py \
    --train-data Datasets/train.csv \
    --val-data Datasets/validation.csv \
    --batch-size 2 \
    --epochs 2 \
    --sample-size 100 \
    --output-dir "$OUTPUT_DIR" \
    > "$OUTPUT_DIR/logs/training.log" 2>&1 &

TRAIN_PID=$!
echo $TRAIN_PID > "$OUTPUT_DIR/train.pid"

echo "✅ 训练已启动！"
echo ""
echo "📊 监控命令："
echo "================================"
echo ""
echo "1) 查看进程状态:"
echo "   ps aux | grep $TRAIN_PID"
echo ""
echo "2) 查看GPU使用:"
echo "   nvidia-smi"
echo ""
echo "3) 查看训练日志:"
echo "   tail -f $OUTPUT_DIR/logs/training.log"
echo ""
echo "4) 完整监控:"
echo "   ./monitor_all.sh"
echo ""
echo "5) 停止训练:"
echo "   kill $TRAIN_PID"
echo ""
echo "================================"

# 等待3秒后显示初始状态
sleep 3

echo ""
echo "🔍 初始状态检查："
if kill -0 $TRAIN_PID 2>/dev/null; then
    echo "✅ 训练进程运行中 (PID: $TRAIN_PID)"
    echo ""
    echo "📝 最新日志:"
    tail -5 "$OUTPUT_DIR/logs/training.log" 2>/dev/null || echo "等待日志生成..."
else
    echo "❌ 训练进程启动失败"
    echo "查看错误: tail -20 $OUTPUT_DIR/logs/training.log"
fi