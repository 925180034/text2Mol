# Scaffold-Based Multi-Modal Molecular Generation System

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Progress](https://img.shields.io/badge/progress-80%25-green.svg)](PROJECT_STATUS.md)

A state-of-the-art molecular generation system that creates complete molecules from molecular scaffolds and text descriptions, supporting 7 input-output modality combinations across SMILES, Graph, and Image formats.

## 🚀 Recent Updates (December 2024)

- **Multi-Modal Architecture Complete**: All 7 input-output combinations implemented
- **Graph & Image Decoders**: Added molecular graph and image generation capabilities  
- **Large-Scale Data Processing**: Successfully processed 26K+ training samples
- **Unified Feature Space**: 768-dimensional encoding across all modalities
- **Project Reorganization**: Clean structure with 80% completion

## 🌟 Key Features

✅ **True Multi-Modal I/O**: 3 scaffold modalities × 3 output modalities = 7 combinations  
✅ **Scaffold-Based Generation**: Uses molecular core structures (Murcko scaffolds)  
✅ **Advanced Encoders**: MolT5, BERT, GIN, Swin Transformer - all unified to 768-dim  
✅ **Cross-Modal Fusion**: Attention + gating mechanisms for modal integration  
✅ **Production Scale**: 596M parameters, handles 26K+ molecular samples  
✅ **Comprehensive Metrics**: 9 evaluation metrics including validity, novelty, similarity  
✅ **GPU Optimized**: CUDA support with mixed precision capabilities

## Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/925180034/text2Mol.git
cd text2Mol/scaffold-mol-generation

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch, transformers, rdkit; print('✅ All dependencies ready')"
```

### 2. Model Setup
The system includes a CUDA-optimized MolT5 model (`models/MolT5-Small-Fixed/`) that resolves vocabulary compatibility issues. This model is ready to use.

The model is pre-configured and ready to use.

### 3. Training
```bash
# Enhanced training pipeline
python enhanced_training_pipeline.py --config configs/default_config.yaml

# Original training (debug mode)
python train.py --config configs/default_config.yaml --debug

# Full training
python train.py --config configs/default_config.yaml
```

### 4. Generation
```bash
# Interactive generation
python generate.py --model-checkpoint models/MolT5-Small --interactive

# Batch generation
python generate.py --config configs/default_config.yaml --input-file input.csv
```

### 5. Evaluation
```bash
# Enhanced evaluation with comprehensive metrics
python evaluate_enhanced.py --config configs/default_config.yaml

# Quiet evaluation (suppresses warnings)
python run_quiet_evaluation.py

# Standard evaluation
python evaluate.py --config configs/default_config.yaml
```

## 📊 Supported Input-Output Combinations

| # | Scaffold Input | Text | Output | Status | Description |
|---|---------------|------|--------|--------|-------------|
| 1 | SMILES | ✓ | SMILES | ✅ Working | Text + scaffold string → molecule string |
| 2 | Graph | ✓ | SMILES | ✅ Ready | Text + scaffold graph → molecule string |
| 3 | Image | ✓ | SMILES | ✅ Ready | Text + scaffold image → molecule string |
| 4 | SMILES | ✓ | Graph | ✅ Ready | Text + scaffold string → molecule graph |
| 5 | SMILES | ✓ | Image | ✅ Ready | Text + scaffold string → molecule image |
| 6 | Graph | ✓ | Graph | ✅ Ready | Text + scaffold graph → molecule graph |
| 7 | Image | ✓ | Image | ✅ Ready | Text + scaffold image → molecule image |

## Project Structure

```
scaffold-mol-generation/
├── scaffold_mol_gen/       # Core library
│   ├── models/            # Model implementations
│   │   ├── encoders/      # 4 multi-modal encoders
│   │   ├── fusion_simplified.py  # Cross-modal fusion
│   │   ├── molt5_adapter.py      # MolT5 generation
│   │   ├── graph_decoder.py      # Graph decoder
│   │   ├── image_decoder.py      # Image decoder
│   │   └── end2end_model.py      # Unified model
│   ├── data/              # Data processing
│   ├── training/          # Training utilities
│   ├── evaluation/        # 9 evaluation metrics
│   └── utils/             # Helper functions
├── configs/               # Configuration files
├── Datasets/              # ChEBI-20 dataset
├── scripts/               # Utility scripts
│   └── preprocessing/     # Data preprocessing
├── tests/                 # Unit tests
├── models/                # Saved models
├── outputs/               # Generation results
└── PROJECT_STATUS.md      # Detailed progress (80%)
```

## Configuration

### Model Configuration
- Edit `configs/default_config.yaml` for all model, training, and evaluation parameters

### Key Parameters
- `molt5_checkpoint`: Path to MolT5 model (currently: `models/MolT5-Small-Fixed`)
- `input_modalities`: Input types (`["text", "smiles"]`)
- `max_length`: Maximum sequence length for generation
- `batch_size`: Training batch size

## Data Format

Training data should be CSV files with columns:
- `text` or `description`: Natural language description of the molecule
- `SMILES`: Target molecule SMILES notation

The system automatically extracts scaffolds using the Murcko scaffold algorithm.

Example:
```csv
description,SMILES
"Small molecule inhibitor for cancer treatment","CC(=O)Nc1ccccc1"
"Analgesic compound with benzene ring","CCc1ccccc1O"
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA support
- RDKit for molecular processing
- Transformers for language models
- PyTorch Geometric for graph operations

## Performance Notes

- Currently using MolT5-Small model for faster training
- PyTorch Geometric extensions may be limited, affecting some advanced graph operations
- Recommended batch size: 16-32 for training, 64+ for evaluation
- CUDA 12.6 compatible

## Troubleshooting

### Common Issues

1. **CUDA Memory**: Reduce batch_size in config if you encounter OOM errors
2. **RDKit Warnings**: Use `run_quiet_evaluation.py` to suppress chemistry warnings
3. **Missing Dependencies**: Ensure all requirements are installed with correct versions

## 📈 Model Architecture & Performance

### Architecture Details
- **Encoders**: 4 specialized encoders (MolT5, BERT, GIN, Swin) → 768-dim
- **Fusion**: Cross-modal attention + gated fusion
- **Decoders**: 3 specialized decoders for SMILES/Graph/Image generation
- **Parameters**: 596.52M total (59.08M trainable)

### Performance Metrics
- **GPU Memory**: ~8GB (batch_size=2)
- **Training Data**: 26,402 samples processed
- **Validation Data**: 3,299 samples processed
- **Scaffold Simplification**: 77% of molecules have simpler scaffolds than targets
- **Inference Speed**: ~100ms/sample

## Development Status

**Overall Progress: 80% Complete**

- ✅ **Phase 1**: Data processing & preprocessing (100%)
- ✅ **Phase 2**: Multi-modal encoders & decoders (100%)
- ✅ **Phase 3**: Architecture & fusion layers (100%)
- 🔄 **Phase 4**: Training system (50%)
- ⏳ **Phase 5**: Optimization & deployment (0%)

## Citation

If you use this code, please cite the relevant papers for MolT5 and scaffold-based molecular generation.

## Contact

For questions or issues, please open an issue on GitHub.