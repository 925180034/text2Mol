# Training Scripts Documentation

## Overview
This directory contains the official training scripts for the progressive two-stage training approach. All outdated and experimental scripts have been archived.

## Stage 1: Multi-Modal Alignment Training

### Main Scripts
1. **train_stage1_alignment.py**
   - Base implementation of Stage 1 alignment training
   - Implements contrastive learning across modalities
   - Configuration: batch_size=32, epochs=50, lr=1e-4

2. **train_stage1_optimized.py** ✅ (USED)
   - Optimized version for 32GB GPU
   - Memory-efficient implementation
   - Configuration: batch_size=8, epochs=50, lr=1e-4
   - **Result**: Validation loss 1.3598

3. **launch_stage1_training.sh**
   - Shell script to launch Stage 1 training
   - Handles output directory creation and logging
   - Sets up proper environment

## Stage 2: SMILES Generation Training

### Main Scripts
1. **train_stage2_generation.py**
   - Base implementation of Stage 2 generation
   - Supports all 3 input modalities → SMILES
   - Configuration: batch_size=16, epochs=100, lr=5e-5

2. **train_stage2_simplified.py** ✅ (USED)
   - Simplified and optimized version
   - Better memory management and stability
   - Configuration: batch_size=4, epochs=30, lr=5e-5
   - **Result**: Validation loss 1.0277

3. **launch_stage2_training.sh**
   - Shell script to launch Stage 2 training
   - Manages checkpoint loading from Stage 1
   - Handles logging and monitoring

## Usage Instructions

### Stage 1 Training
```bash
# Using optimized version (recommended)
python train_stage1_optimized.py \
    --train-data Datasets/train.csv \
    --val-data Datasets/validation.csv \
    --batch-size 8 \
    --epochs 50 \
    --output-dir /root/autodl-tmp/text2Mol-stage1

# Or using launch script
./launch_stage1_training.sh
```

### Stage 2 Training
```bash
# Using simplified version (recommended)
python train_stage2_simplified.py \
    --train-data Datasets/train.csv \
    --val-data Datasets/validation.csv \
    --stage1-checkpoint /root/autodl-tmp/text2Mol-stage1/best_model.pt \
    --batch-size 4 \
    --epochs 30 \
    --output-dir /root/autodl-tmp/text2Mol-stage2

# Or using launch script
./launch_stage2_training.sh
```

## Utility Scripts
Located in `utils/` directory:
- **monitor_training.py**: General training monitoring
- **monitor_stage2_training.py**: Stage 2 specific monitoring

## Archived Scripts
Moved to `archive_outdated_training/`:
- train_stage2_nine_modality.py (problematic 9-modality attempt)
- test_training_stability.py (testing script)
- quick_start.sh (referenced non-existent files)

## Important Notes

### Progressive Training is MANDATORY
- Do NOT attempt to train all 9 modalities simultaneously
- Complete Stage 1 before starting Stage 2
- Use Stage 1 checkpoint to initialize Stage 2

### Resource Requirements
- Stage 1: ~8GB GPU memory
- Stage 2: ~8GB GPU memory
- Training time: 2-4 hours per stage

### Completed Training Results
- **Stage 1**: `/root/autodl-tmp/text2Mol-stage1/best_model.pt` (2.2GB)
- **Stage 2**: `/root/autodl-tmp/text2Mol-stage2/best_model_stage2.pt` (3.6GB)

## Next Steps
After training completion:
1. Evaluate using `nine_modality_evaluation_fixed.py`
2. Test generation quality with metrics
3. Proceed to Stage 3 (Graph Decoder) development

---
*Last Updated: 2025-08-17*  
*Cleaned and organized training scripts*