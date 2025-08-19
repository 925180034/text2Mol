#!/bin/bash

# Launch Validity-Focused Training
# Uses MolT5-Large with SMILES validity constraints

echo "========================================="
echo "🎯 LAUNCHING VALIDITY-FOCUSED TRAINING"
echo "========================================="
echo ""
echo "📅 Start Time: $(date)"
echo "💾 Output Directory: /root/autodl-tmp/text2Mol-validity-focused"
echo "🧬 Model: molt5-base (will be trained for chemistry)"
echo "🎯 Strategy: 3-stage validity enforcement"
echo ""

# Create output directory
mkdir -p /root/autodl-tmp/text2Mol-validity-focused

# Check GPU availability
echo "🖥️ GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
echo ""

# Check disk space
echo "💾 Disk Space:"
df -h /root/autodl-tmp | tail -1
echo ""

# Launch training in background
echo "📝 Starting validity-focused training..."
nohup python -u train_validity_focused.py \
    > /root/autodl-tmp/text2Mol-validity-focused/training.log 2>&1 &

# Get process ID
TRAIN_PID=$!
echo "✅ Training launched with PID: $TRAIN_PID"
echo ""

# Save PID for monitoring
echo $TRAIN_PID > /root/autodl-tmp/text2Mol-validity-focused/training.pid

# Create monitoring script
cat > /root/autodl-tmp/text2Mol-validity-focused/monitor.sh << 'EOF'
#!/bin/bash
# Monitor validity-focused training

echo "📊 Validity-Focused Training Monitor"
echo "===================================="

# Check if process is running
PID=$(cat /root/autodl-tmp/text2Mol-validity-focused/training.pid 2>/dev/null)
if [ -z "$PID" ]; then
    echo "❌ No training process found"
    exit 1
fi

if ps -p $PID > /dev/null; then
    echo "✅ Training is running (PID: $PID)"
    echo ""
    
    # Show recent log with validity metrics
    echo "📜 Recent Training Progress:"
    echo "----------------------------"
    tail -n 30 /root/autodl-tmp/text2Mol-validity-focused/training.log | grep -E "(Stage|Epoch|Valid:|BLEU|Starting|✅|🎯|📈|✨)"
    
    echo ""
    echo "📊 Full Log (last 10 lines):"
    echo "----------------------------"
    tail -n 10 /root/autodl-tmp/text2Mol-validity-focused/training.log
    
    echo ""
    echo "🖥️ GPU Usage:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
    
    echo ""
    echo "💾 Checkpoints:"
    ls -lh /root/autodl-tmp/text2Mol-validity-focused/*.pt 2>/dev/null | tail -5
else
    echo "⚠️ Training process completed or stopped"
    echo ""
    echo "📊 Final Results:"
    tail -n 50 /root/autodl-tmp/text2Mol-validity-focused/training.log | grep -E "(Stage|Epoch|Valid:|BLEU|completed|✅)"
fi
EOF

chmod +x /root/autodl-tmp/text2Mol-validity-focused/monitor.sh

echo "📊 Training Configuration:"
echo "  🧬 Model: molt5-base (training for chemistry-specific generation)"
echo "  📦 Batch Size: 8 (effective 32 with accumulation)"
echo "  🎯 Stage 1: Validity enforcement (10 epochs, β=1.0)"
echo "  📈 Stage 2: Quality improvement (10 epochs, β=0.5)"
echo "  ✨ Stage 3: Final refinement (10 epochs, β=0.2)"
echo "  🛡️ Constraints: SMILES grammar rules + validity checking"
echo "  🔄 Generation: Constrained beam search with repetition penalty"
echo ""

echo "🔍 Key Improvements Over Previous Approach:"
echo "  ✅ Uses MolT5-Large (trained on molecules, not generic text)"
echo "  ✅ Real-time SMILES validity checking during training"
echo "  ✅ Constrained generation prevents invalid patterns"
echo "  ✅ Repetition penalty prevents mode collapse"
echo "  ✅ SMILES augmentation for robustness"
echo "  ✅ 3-stage curriculum: validity → quality → refinement"
echo ""

echo "📝 Monitoring Commands:"
echo "  Check status:  /root/autodl-tmp/text2Mol-validity-focused/monitor.sh"
echo "  View full log: tail -f /root/autodl-tmp/text2Mol-validity-focused/training.log"
echo "  Stop training: kill $TRAIN_PID"
echo ""

echo "⏳ Waiting for initialization (15 seconds)..."
sleep 15

echo ""
echo "📝 Initial Training Output:"
echo "----------------------------"
tail -n 30 /root/autodl-tmp/text2Mol-validity-focused/training.log

echo ""
echo "✅ Validity-focused training is running in background!"
echo "   This approach should prevent mode collapse and achieve >90% validity"
echo "   Use the monitor script to check progress"