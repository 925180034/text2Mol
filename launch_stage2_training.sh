#!/bin/bash

# Launch Stage 2 Nine-Modality Training with Monitoring
# Optimized for 32GB GPU with background execution

echo "=========================================="
echo "🚀 LAUNCHING STAGE 2 NINE-MODALITY TRAINING"
echo "=========================================="
echo ""

# Configuration
OUTPUT_DIR="/root/autodl-tmp/text2Mol-stage2"
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

# Check Stage 1 model exists
STAGE1_MODEL="/root/autodl-tmp/text2Mol-stage1/best_model.pt"
if [ -f "$STAGE1_MODEL" ]; then
    echo "✅ Stage 1 model found: $STAGE1_MODEL"
    echo "   Size: $(du -sh $STAGE1_MODEL | cut -f1)"
else
    echo "❌ Stage 1 model not found!"
    echo "   Expected at: $STAGE1_MODEL"
    echo "   Please complete Stage 1 training first."
    exit 1
fi
echo ""

# Activate conda environment if needed
if [ -d "/root/miniconda3" ]; then
    source /root/miniconda3/bin/activate text2Mol
    if [ -n "$CONDA_DEFAULT_ENV" ]; then
        echo "✅ Conda environment: $CONDA_DEFAULT_ENV"
    fi
fi

# Display training configuration
echo "🎯 Training Configuration:"
echo "   - 9 Modality Combinations (3×3 matrix)"
echo "   - Input: SMILES, Graph, Image"
echo "   - Output: SMILES, Graph, Image"
echo "   - Stage 1 weights: Loaded"
echo "   - Storage limit: 50GB"
echo "   - GPU memory: 32GB"
echo ""

# Launch training in background
echo "🚀 Starting training in background..."
echo "Output directory: $OUTPUT_DIR"
echo ""

cd /root/text2Mol/scaffold-mol-generation
nohup python train_stage2_nine_modality.py > $LOG_FILE 2>&1 &
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
    echo "1. Real-time 9-modality monitor (recommended):"
    echo "   python monitor_stage2_training.py"
    echo ""
    echo "2. Watch log file:"
    echo "   tail -f $OUTPUT_DIR/training_*.log"
    echo ""
    echo "3. Check GPU usage:"
    echo "   watch -n 2 nvidia-smi"
    echo ""
    echo "4. Check modality grid performance:"
    echo "   grep 'Modality Performance' -A 10 $OUTPUT_DIR/training_*.log"
    echo ""
    echo "5. Stop training:"
    echo "   kill $TRAIN_PID"
    echo ""
    echo "=========================================="
    echo ""
    
    # Ask if user wants to start monitor
    read -p "Start 9-modality monitor now? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Starting monitor..."
        python monitor_stage2_training.py
    else
        echo "You can start the monitor later with:"
        echo "python monitor_stage2_training.py"
    fi
else
    echo "❌ Training failed to start!"
    echo "Check the log file for errors:"
    echo "cat $LOG_FILE"
    exit 1
fi