#!/bin/bash

# Launch Simple Validity Training
# Focus on getting valid SMILES generation first

echo "========================================="
echo "🎯 LAUNCHING SIMPLE VALIDITY TRAINING"
echo "========================================="
echo ""
echo "📅 Start Time: $(date)"
echo "💾 Output Directory: /root/autodl-tmp/text2Mol-simple-valid"
echo "🧬 Model: molt5-base"
echo "🎯 Goal: Achieve >90% SMILES validity"
echo ""

# Create output directory
mkdir -p /root/autodl-tmp/text2Mol-simple-valid

# Check GPU
echo "🖥️ GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
echo ""

# Launch training
echo "📝 Starting simple validity training..."
nohup python -u train_simple_valid.py \
    > /root/autodl-tmp/text2Mol-simple-valid/training.log 2>&1 &

TRAIN_PID=$!
echo "✅ Training launched with PID: $TRAIN_PID"
echo ""

# Save PID
echo $TRAIN_PID > /root/autodl-tmp/text2Mol-simple-valid/training.pid

# Create monitor script
cat > /root/autodl-tmp/text2Mol-simple-valid/monitor.sh << 'EOF'
#!/bin/bash
echo "📊 Simple Training Monitor"
echo "=========================="

PID=$(cat /root/autodl-tmp/text2Mol-simple-valid/training.pid 2>/dev/null)
if [ -z "$PID" ]; then
    echo "❌ No training process found"
    exit 1
fi

if ps -p $PID > /dev/null; then
    echo "✅ Training is running (PID: $PID)"
    echo ""
    echo "📜 Recent Progress:"
    tail -n 20 /root/autodl-tmp/text2Mol-simple-valid/training.log | grep -E "(Epoch|Loss|Validity|Saved|Achieved|Starting|completed)"
    echo ""
    echo "🖥️ GPU Usage:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
else
    echo "⚠️ Training completed or stopped"
    echo ""
    echo "📊 Final Results:"
    tail -n 30 /root/autodl-tmp/text2Mol-simple-valid/training.log | grep -E "(Best|Epoch|Validity|completed)"
fi
EOF

chmod +x /root/autodl-tmp/text2Mol-simple-valid/monitor.sh

echo "📊 Training Configuration:"
echo "  Model: molt5-base"
echo "  Batch Size: 16"
echo "  Learning Rate: 5e-5"
echo "  Epochs: 20 (early stop at 90% validity)"
echo "  Focus: Simple greedy generation to avoid CUDA errors"
echo ""

echo "🔍 Commands:"
echo "  Monitor: /root/autodl-tmp/text2Mol-simple-valid/monitor.sh"
echo "  Log: tail -f /root/autodl-tmp/text2Mol-simple-valid/training.log"
echo "  Stop: kill $TRAIN_PID"
echo ""

sleep 10
echo "📝 Initial output:"
tail -n 20 /root/autodl-tmp/text2Mol-simple-valid/training.log

echo ""
echo "✅ Simple training running in background!"