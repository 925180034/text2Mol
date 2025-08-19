#!/bin/bash

# Launch Stage 1 Alignment Training with Monitoring
# Optimized for 32GB GPU with background execution

echo "=========================================="
echo "🚀 LAUNCHING STAGE 1 ALIGNMENT TRAINING"
echo "=========================================="
echo ""

# Configuration
OUTPUT_DIR="/root/autodl-tmp/text2Mol-stage1"
LOG_FILE="$OUTPUT_DIR/launch.log"
PID_FILE="$OUTPUT_DIR/training.pid"

# Create output directory
mkdir -p $OUTPUT_DIR

# Check if training is already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat $PID_FILE)
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "⚠️ Training is already running with PID $OLD_PID"
        echo "Stop it first with: kill $OLD_PID"
        exit 1
    else
        echo "Removing stale PID file"
        rm $PID_FILE
    fi
fi

# Check GPU availability
echo "🖥️ Checking GPU status..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Check disk space
echo "💾 Checking disk space..."
df -h /root/autodl-tmp | tail -1
echo ""

# Activate conda environment if needed
if [ -d "/root/miniconda3" ]; then
    source /root/miniconda3/bin/activate
    if [ -n "$CONDA_DEFAULT_ENV" ]; then
        echo "✅ Conda environment: $CONDA_DEFAULT_ENV"
    fi
fi

# Launch training in background
echo "🎯 Starting training in background..."
echo "Output directory: $OUTPUT_DIR"
echo ""

nohup python train_stage1_optimized.py > $LOG_FILE 2>&1 &
TRAIN_PID=$!

# Save PID
echo $TRAIN_PID > $PID_FILE

echo "✅ Training started with PID: $TRAIN_PID"
echo "📝 Log file: $LOG_FILE"
echo ""

# Wait a moment for training to initialize
sleep 5

# Check if training is still running
if ps -p $TRAIN_PID > /dev/null; then
    echo "✅ Training is running successfully!"
    echo ""
    echo "=========================================="
    echo "📊 MONITORING OPTIONS:"
    echo "=========================================="
    echo ""
    echo "1. Real-time monitor (recommended):"
    echo "   python monitor_training.py"
    echo ""
    echo "2. Watch log file:"
    echo "   tail -f $OUTPUT_DIR/training_*.log"
    echo ""
    echo "3. Check GPU usage:"
    echo "   watch -n 2 nvidia-smi"
    echo ""
    echo "4. Stop training:"
    echo "   kill $TRAIN_PID"
    echo ""
    echo "=========================================="
    echo ""
    
    # Ask if user wants to start monitor
    read -p "Start real-time monitor now? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Starting monitor..."
        python monitor_training.py
    else
        echo "You can start the monitor later with:"
        echo "python monitor_training.py"
    fi
else
    echo "❌ Training failed to start!"
    echo "Check the log file for errors:"
    echo "cat $LOG_FILE"
    exit 1
fi