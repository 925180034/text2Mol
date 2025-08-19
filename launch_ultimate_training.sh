#!/bin/bash

# Launch Ultimate Stage 2 Training Script
# Comprehensive training with curriculum learning and teacher forcing
# Optimized for 32GB GPU

echo "=================================="
echo "🚀 LAUNCHING ULTIMATE STAGE 2 TRAINING"
echo "=================================="
echo ""
echo "📅 Start Time: $(date)"
echo "💾 Output Directory: /root/autodl-tmp/text2Mol-ultimate-stage2"
echo "🎯 Training Strategy: 4-phase curriculum learning with teacher forcing"
echo "⚙️ GPU Configuration: Optimized for 32GB memory"
echo ""

# Create output directory
mkdir -p /root/autodl-tmp/text2Mol-ultimate-stage2

# Check GPU availability
echo "🖥️ GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
echo ""

# Launch training in background
echo "📝 Starting training process..."
nohup python -u train_stage2_ultimate.py \
    > /root/autodl-tmp/text2Mol-ultimate-stage2/training.log 2>&1 &

# Get process ID
TRAIN_PID=$!
echo "✅ Training launched with PID: $TRAIN_PID"
echo ""

# Save PID for monitoring
echo $TRAIN_PID > /root/autodl-tmp/text2Mol-ultimate-stage2/training.pid

# Create monitoring script
cat > /root/autodl-tmp/text2Mol-ultimate-stage2/monitor.sh << 'EOF'
#!/bin/bash
# Monitor training progress

echo "📊 Training Progress Monitor"
echo "============================"

# Check if process is running
PID=$(cat /root/autodl-tmp/text2Mol-ultimate-stage2/training.pid 2>/dev/null)
if [ -z "$PID" ]; then
    echo "❌ No training process found"
    exit 1
fi

if ps -p $PID > /dev/null; then
    echo "✅ Training is running (PID: $PID)"
    echo ""
    
    # Show last 20 lines of log
    echo "📜 Recent Log Output:"
    echo "--------------------"
    tail -n 20 /root/autodl-tmp/text2Mol-ultimate-stage2/training.log
    
    echo ""
    echo "🖥️ GPU Usage:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
    
    echo ""
    echo "💾 Checkpoints:"
    ls -lh /root/autodl-tmp/text2Mol-ultimate-stage2/*.pt 2>/dev/null | tail -5
else
    echo "⚠️ Training process completed or stopped"
    echo "Check the full log at: /root/autodl-tmp/text2Mol-ultimate-stage2/training.log"
fi
EOF

chmod +x /root/autodl-tmp/text2Mol-ultimate-stage2/monitor.sh

echo "📊 Training Configuration:"
echo "  - Batch Size: 16 (effective 32 with gradient accumulation)"
echo "  - Learning Rate: 5e-5"
echo "  - Total Epochs: 50 (10+10+15+15 across 4 phases)"
echo "  - Teacher Forcing: Starting at 100%, decaying to 30%"
echo "  - Curriculum: Easy → Medium → Hard → All examples"
echo ""

echo "🔍 Monitoring Commands:"
echo "  - Check status: /root/autodl-tmp/text2Mol-ultimate-stage2/monitor.sh"
echo "  - View full log: tail -f /root/autodl-tmp/text2Mol-ultimate-stage2/training.log"
echo "  - Stop training: kill $TRAIN_PID"
echo ""

echo "📝 Initial log output (first 10 seconds):"
echo "----------------------------------------"
sleep 10
tail -n 30 /root/autodl-tmp/text2Mol-ultimate-stage2/training.log

echo ""
echo "✅ Training is running in background!"
echo "   Use the monitor script to check progress"