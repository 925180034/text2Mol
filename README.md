# Scaffold-Based Molecular Generation

A state-of-the-art system for generating molecules while preserving molecular scaffolds using multi-modal inputs.

## Features

- **Multi-Modal Input**: Text descriptions, SMILES strings, molecular graphs, and images
- **Scaffold Preservation**: Maintains molecular scaffolds during generation
- **MolT5 Integration**: Built on pre-trained MolT5 transformer models
- **Advanced Fusion**: Cross-modal attention and gating mechanisms
- **Comprehensive Evaluation**: Multiple molecular property metrics

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
**Important**: Models are not included in the repository due to size constraints.

```python
# Quick model download
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "laituan245/molt5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

tokenizer.save_pretrained("models/MolT5-Small")
model.save_pretrained("models/MolT5-Small")
```

For detailed setup instructions, see: [MODEL_SETUP.md](MODEL_SETUP.md)

### 3. Training
```bash
# Start training with debug mode
python train.py --config configs/training_config.yaml --debug

# Full training
python train.py --config configs/training_config.yaml
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
python evaluate.py --config configs/evaluation_config.yaml
```

## Project Structure

```
scaffold-mol-generation/
├── configs/                 # Configuration files  
├── Datasets/               # Training/validation data
├── models/                 # Pre-trained models
├── scaffold_mol_gen/       # Core package
│   ├── data/              # Dataset handling
│   ├── models/            # Model architectures  
│   ├── training/          # Training utilities
│   ├── evaluation/        # Evaluation metrics
│   ├── utils/             # Helper functions
│   └── api/               # Interactive API
├── train.py               # Training script
├── generate.py            # Generation script
└── evaluate.py            # Evaluation script
```

## Configuration

### Model Configuration
- Edit `configs/default_config.yaml` for model architecture
- Modify `configs/training_config.yaml` for training parameters
- Adjust `configs/evaluation_config.yaml` for evaluation metrics

### Key Parameters
- `molt5_checkpoint`: Path to MolT5 model (currently: `models/MolT5-Small`)
- `input_modalities`: Input types (`["text", "smiles"]`)
- `max_length`: Maximum sequence length for generation
- `batch_size`: Training batch size

## Data Format

Training data should be CSV files with columns:
- `text`: Natural language description
- `SMILES`: Target molecule SMILES
- `scaffold_smiles`: Scaffold SMILES (optional)

Example:
```csv
text,SMILES,scaffold_smiles
"Design analgesic with benzene ring","CC(=O)Nc1ccccc1","c1ccccc1"
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

## Citation

If you use this code, please cite the relevant papers for MolT5 and scaffold-based molecular generation.