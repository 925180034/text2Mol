# Scaffold-Based Molecular Generation

A production-ready system for scaffold-based molecular generation using multi-modal inputs with CUDA-optimized MolT5 integration.

## 🚀 Recent Updates (August 2025)

- **Enhanced Evaluation Pipeline**: New comprehensive evaluation system with detailed metrics
- **Optimized Training**: Improved training pipeline with better convergence
- **Extended I/O**: Support for multiple input/output formats
- **Optimized Prompting**: Advanced prompting strategies for better generation
- **Code Cleanup**: Organized project structure with archived legacy files

## Features

✅ **Multi-Modal Input**: Text descriptions + SMILES sequences  
✅ **CUDA Optimized**: Fixed vocabulary compatibility for GPU training  
✅ **Dual Tokenizer Architecture**: SciBERT for text, T5 for molecular data  
✅ **Real Dataset Ready**: Tested with 19,795+ training samples  
✅ **Production Model**: 454M parameters with scaffold preservation  
✅ **Advanced Fusion**: Cross-modal attention and gating mechanisms
✅ **Enhanced Evaluation**: Comprehensive metrics with visualization support
✅ **Optimized Prompting**: Improved generation quality through prompt engineering

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

## Project Structure

```
scaffold-mol-generation/
├── configs/                 # Configuration files
│   └── default_config.yaml  # Main configuration
├── Datasets/               # Training/validation data
├── models/                 # Pre-trained models
├── scaffold_mol_gen/       # Core package
│   ├── data/              # Dataset handling
│   │   ├── dataset.py
│   │   └── dual_tokenizer_dataset.py
│   ├── models/            # Model architectures
│   ├── training/          # Training utilities
│   ├── evaluation/        # Evaluation metrics
│   ├── utils/             # Helper functions
│   │   └── graph_utils.py
│   └── api/               # Interactive API
├── Core Scripts:
│   ├── train.py                    # Training script
│   ├── generate.py                 # Generation script
│   ├── evaluate.py                 # Evaluation script
│   ├── evaluate_enhanced.py        # Enhanced evaluation
│   ├── enhanced_training_pipeline.py # Improved training
│   ├── extended_input_output.py    # Extended I/O support
│   ├── optimized_prompting.py      # Prompt optimization
│   └── comprehensive_validation.py  # Validation suite
├── Runners:
│   ├── run_your_model_evaluation.py
│   ├── run_enhanced_model_evaluation.py
│   └── run_quiet_evaluation.py
└── Results:
    ├── evaluation_results_enhanced/
    └── your_model_evaluation_results/
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

## Development Status

- ✅ Phase 1: Core implementation complete
- ✅ Phase 2: Enhanced evaluation and metrics
- 🔄 Phase 3: Optimization and scaling (in progress)
- ⏳ Phase 4: Production deployment (planned)

## Citation

If you use this code, please cite the relevant papers for MolT5 and scaffold-based molecular generation.

## Contact

For questions or issues, please open an issue on GitHub.