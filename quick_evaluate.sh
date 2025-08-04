#!/bin/bash

# Quick evaluation script for Phase 1 enhanced metrics
# Usage: ./quick_evaluate.sh [model_checkpoint_path] [num_samples]

MODEL_CHECKPOINT=${1:-"/path/to/your/model/checkpoint"}
NUM_SAMPLES=${2:-500}
OUTPUT_DIR="evaluation_results_$(date +%Y%m%d_%H%M%S)"

echo "🚀 Starting Enhanced Model Evaluation"
echo "Model: $MODEL_CHECKPOINT"
echo "Samples: $NUM_SAMPLES"
echo "Output: $OUTPUT_DIR"
echo ""

# Run enhanced evaluation
python run_enhanced_model_evaluation.py \
    --model-checkpoint "$MODEL_CHECKPOINT" \
    --test-data "Datasets/test.csv" \
    --output-dir "$OUTPUT_DIR" \
    --num-samples $NUM_SAMPLES \
    --batch-size 16 \
    --generation-config "generation_config.json" \
    --save-predictions \
    --device auto

echo ""
echo "✅ Evaluation completed!"
echo "📁 Results saved to: $OUTPUT_DIR"
echo ""
echo "📊 Key files:"
echo "  - $OUTPUT_DIR/evaluation_metrics.json"
echo "  - $OUTPUT_DIR/evaluation_summary.md"
echo "  - $OUTPUT_DIR/predictions_comparison.csv"