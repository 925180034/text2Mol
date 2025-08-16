# Text2Mol: Scaffold-Based Multi-Modal Molecular Generation System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![RDKit](https://img.shields.io/badge/RDKit-2023.03+-green.svg)](https://www.rdkit.org/)
[![Status](https://img.shields.io/badge/status-active-success.svg)](PROJECT_IMPLEMENTATION_GUIDE.md)
[![Progress](https://img.shields.io/badge/progress-Framework_Complete-brightgreen.svg)](docs/NINE_MODALITY_EVALUATION.md)

A state-of-the-art deep learning system for generating complete molecules from molecular scaffolds and text descriptions, supporting **9 input-output modality combinations** across SMILES, Graph, and Image formats.

**Last Updated**: 2025-08-16 | **Version**: 1.0.0

## 🎯 Project Overview

Text2Mol implements a sophisticated two-stage training approach inspired by GIT-Mol, enabling flexible molecular generation across multiple modalities. The system first learns cross-modal alignments through contrastive learning, then performs instruction-guided generation for all 9 modality combinations.

### Key Innovations
- **Two-Stage Training**: Alignment pre-training → Instruction-guided generation
- **9 Modality Combinations**: Complete coverage of (SMILES/Graph/Image) × (SMILES/Graph/Image)
- **Unified Architecture**: Single model handles all tasks through instruction tokens
- **Comprehensive Evaluation**: 11 metrics including validity, novelty, and fingerprint similarities

## 🌟 Features

✅ **Multi-Modal I/O**: 3 input × 3 output modalities = 9 combinations  
✅ **Advanced Encoders**: MolT5, GIN, Swin Transformer, BERT/SciBERT  
✅ **GIT-Former Fusion**: Cross-attention + gated fusion mechanisms  
✅ **Scaffold-Guided**: Molecular core structure preservation  
✅ **Production Ready**: 596M parameters, optimized for deployment  
✅ **Comprehensive Metrics**: Validity, Uniqueness, Novelty, BLEU, FTS, FCD, etc.

## 📊 Evaluation Metrics

Our system implements 11 state-of-the-art evaluation metrics:

| Metric | Description | Target |
|--------|-------------|--------|
| **Validity** | Chemical validity of generated molecules | >95% |
| **Uniqueness** | Percentage of unique molecules | >90% |
| **Novelty** | Molecules not in training set | >85% |
| **BLEU** | Sequence similarity (1-4 grams) | >0.7 |
| **Exact Match** | Perfect molecular matches | >30% |
| **Levenshtein** | Edit distance between sequences | Lower is better |
| **MACCS FTS** | MACCS fingerprint similarity | >0.6 |
| **Morgan FTS** | Morgan fingerprint similarity | >0.6 |
| **RDK FTS** | RDKit fingerprint similarity | >0.6 |
| **FCD** | Fréchet ChemNet Distance | <5.0 |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/925180034/text2Mol.git
cd text2Mol/scaffold-mol-generation

# Create conda environment
conda create -n text2mol python=3.9
conda activate text2mol

# Install dependencies
pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.30.2
pip install rdkit==2023.03.3
pip install torch-geometric==2.3.1
pip install timm==0.9.2
pip install pandas numpy tqdm tensorboard
pip install nltk levenshtein scipy
```

### Two-Stage Training

#### Stage 1: Multi-Modal Alignment
```bash
python train_stage1_alignment.py \
    --data_path data/chebi20_mm/ \
    --output_dir models/stage1/ \
    --batch_size 32 \
    --num_epochs 50 \
    --learning_rate 1e-4
```

#### Stage 2: Instruction-Guided Generation
```bash
python train_stage2_generation.py \
    --data_path data/chebi20_mm/ \
    --stage1_checkpoint models/stage1/best_model.pt \
    --output_dir models/stage2/ \
    --batch_size 16 \
    --num_epochs 100 \
    --learning_rate 5e-5
```

### Evaluation

```bash
# Evaluate all 9 modality combinations
python nine_modality_evaluation_fixed.py \
    --model-path models/stage2/best_model.pt \
    --num-samples 100 \
    --device cuda
```

## 📁 Project Structure

```
scaffold-mol-generation/
├── scaffold_mol_gen/           # Core library
│   ├── models/                 # Neural network models
│   │   ├── encoders/          # Multi-modal encoders
│   │   ├── fusion_simplified.py # GIT-Former fusion
│   │   ├── molt5_adapter.py   # MolT5 integration
│   │   └── end2end_model.py   # Complete pipeline
│   ├── data/                  # Data processing
│   ├── training/              # Training utilities
│   ├── evaluation/            # Metrics and evaluation
│   │   └── comprehensive_metrics.py # All 11 metrics
│   └── utils/                 # Helper functions
├── train_stage1_alignment.py  # Stage 1 training
├── train_stage2_generation.py # Stage 2 training
├── nine_modality_evaluation.py # Full evaluation
├── PROJECT_IMPLEMENTATION_GUIDE.md # Complete guide
└── docs/                      # Documentation
```

## 🔬 9-Modality Matrix

| Input → Output | SMILES | Graph | Image |
|----------------|--------|-------|-------|
| **SMILES** | ✅ Direct generation | ✅ Structure conversion | ✅ Visualization |
| **Graph** | ✅ Sequence extraction | ✅ Graph transformation | ✅ Graph rendering |
| **Image** | ✅ OCR-like extraction | ✅ Structure parsing | ✅ Image enhancement |

## 📊 Dataset

The system uses the ChEBI-20 dataset with multi-modal annotations:
- **Size**: 33,010 molecules
- **Modalities**: SMILES, molecular graphs, 2D images, text descriptions
- **Splits**: Train (80%), Validation (10%), Test (10%)

## 📚 Documentation

- **[PROJECT_IMPLEMENTATION_GUIDE.md](PROJECT_IMPLEMENTATION_GUIDE.md)** - Complete implementation guide
- **[docs/NINE_MODALITY_EVALUATION.md](docs/NINE_MODALITY_EVALUATION.md)** - Evaluation system documentation
- **[docs/SYSTEM_ARCHITECTURE.md](docs/SYSTEM_ARCHITECTURE.md)** - Technical architecture details

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details.

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{text2mol2025,
  title={Text2Mol: Scaffold-Based Multi-Modal Molecular Generation},
  author={Your Name},
  year={2025},
  url={https://github.com/925180034/text2Mol}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- GIT-Mol for the multi-modal alignment inspiration
- MolT5 for the molecular language model
- ChEBI for the molecular dataset
- RDKit for molecular processing

## 📧 Contact

For questions or collaboration, please open an issue or contact the maintainers.

---

*Developed with ❤️ for advancing molecular generation research*