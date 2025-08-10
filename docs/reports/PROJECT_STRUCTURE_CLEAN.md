# Clean Project Structure

This document describes the cleaned project structure after removing obsolete and problematic files.

## 📁 Core Project Structure

```
scaffold-mol-generation/
├── 🎯 Core Training Scripts (USE THESE)
│   ├── train_fixed_multimodal.py     # ✅ Fixed single-modality training
│   └── train_joint_multimodal.py     # ✅ Joint multi-modal training with alignment
│
├── 📦 scaffold_mol_gen/               # Core library (DO NOT MODIFY)
│   ├── models/                        # Neural network components
│   │   ├── encoders/                  # Multi-modal encoders
│   │   │   ├── multimodal_encoder.py  # Main encoder orchestrator
│   │   │   ├── smiles_encoder.py      # SMILES → 768-dim
│   │   │   ├── text_encoder.py        # Text → 768-dim
│   │   │   ├── graph_encoder.py       # Graph → 768-dim
│   │   │   └── image_encoder.py       # Image → 768-dim
│   │   ├── end2end_model.py          # Main end-to-end model
│   │   ├── fusion_simplified.py       # Modal fusion layer
│   │   ├── molt5_adapter.py          # MolT5 integration
│   │   └── output_decoders.py        # Output decoders
│   ├── data/                         # Data processing
│   │   ├── multimodal_dataset.py     # Dataset classes
│   │   └── multimodal_preprocessor.py # Preprocessing utilities
│   ├── training/                     # Training utilities
│   │   └── metrics.py                # Evaluation metrics
│   └── evaluation/                   # Evaluation tools
│
├── 📊 Datasets/                       # Training data
│   ├── train.csv                     # 26,402 training samples
│   ├── validation.csv                # 3,299 validation samples
│   └── test.csv                      # 3,309 test samples
│
├── ⚙️ configs/                        # Configuration files
│   └── default_config.yaml           # Default training config
│
├── 🧪 tests/                          # Unit tests
│   ├── test_e2e_simple.py            # ✅ End-to-end test (IMPORTANT)
│   └── test_encoders.py              # Encoder tests
│
├── 📚 docs/                           # Documentation
│   └── (various documentation)
│
├── 📄 Key Documentation Files
│   ├── README.md                     # Project overview
│   ├── TRAINING_SOLUTION_REPORT.md   # Training fixes explained
│   └── PROJECT_STRUCTURE_CLEAN.md    # This file
│
└── 🗃️ archive/                        # Old/deprecated files (IGNORE)
```

## ✅ Files to USE

### Training Scripts
- `train_fixed_multimodal.py` - Use this for stable training
- `train_joint_multimodal.py` - Use this for advanced multi-modal training

### Testing
- `tests/test_e2e_simple.py` - Always run this to verify system works
- `tests/test_encoders.py` - Test individual encoders

### Core Module
- Everything in `scaffold_mol_gen/` - This is the core library

## ❌ Files REMOVED (No longer exist)

### Problematic Training Scripts (DELETED)
- ~~train.py~~ - Had tokenizer issues
- ~~train_multimodal.py~~ - Old problematic version
- ~~enhanced_training_pipeline.py~~ - Over-complex and buggy

### Diagnostic Scripts (DELETED - no longer needed)
- ~~diagnose_training_issue.py~~
- ~~diagnose_molt5.py~~
- ~~trained_models_evaluation.py~~

### Experimental Files (DELETED - cluttered the project)
- ~~fixed_experiment_evaluation.py~~
- ~~debug_real_evaluation.py~~
- ~~complete_evaluation_fixed.py~~
- ~~nine_modality_evaluation_*.py~~
- ~~test_*.py~~ (various test files)
- ~~monitor_*.sh~~ (monitoring scripts)

## 🚀 Quick Start Commands

### 1. Verify Installation
```bash
python tests/test_e2e_simple.py
```

### 2. Train a Model
```bash
# Option A: Fixed training (recommended)
python train_fixed_multimodal.py \
    --batch-size 4 \
    --epochs 5 \
    --lr 5e-5 \
    --output-dir /root/autodl-tmp/text2Mol-outputs/my_training

# Option B: Joint multi-modal training
python train_joint_multimodal.py \
    --batch-size 4 \
    --epochs 5 \
    --lr 5e-5 \
    --alignment-weight 0.1 \
    --output-dir /root/autodl-tmp/text2Mol-outputs/joint_training
```

### 3. Monitor Training
```bash
tensorboard --logdir /root/autodl-tmp/text2Mol-outputs/*/tensorboard
```

## 💡 Important Notes

1. **Only use the two training scripts mentioned** - All others have been deleted
2. **Save models to `/root/autodl-tmp/`** - This is persistent storage
3. **The core library is in `scaffold_mol_gen/`** - Don't modify without understanding
4. **Old files in `archive/` are for reference only** - Don't use them
5. **Training system is fixed** - Tokenizer issues resolved, model paths corrected

## 📊 Model Architecture Summary

```
Input: Scaffold (SMILES/Graph/Image) + Text
           ↓
    Multi-Modal Encoders
           ↓
      768-dim features
           ↓
      Fusion Layer
           ↓
     MolT5 Adapter
           ↓
    SMILES Generation
```

## 🎯 Current Priorities

1. **Train models** using the fixed scripts
2. **Evaluate performance** with proper metrics
3. **Document results** for reproducibility

The project is now clean and ready for effective model training!