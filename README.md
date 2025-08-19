# Text2Mol: Scaffold-Based Multi-Modal Molecular Generation System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![RDKit](https://img.shields.io/badge/RDKit-2023.03+-green.svg)](https://www.rdkit.org/)
[![Status](https://img.shields.io/badge/status-active-success.svg)](PROJECT_IMPLEMENTATION_GUIDE.md)
[![Progress](https://img.shields.io/badge/progress-Framework_Complete-brightgreen.svg)](docs/NINE_MODALITY_EVALUATION.md)

A state-of-the-art deep learning system for generating complete molecules from molecular scaffolds and text descriptions, supporting **9 input-output modality combinations** across SMILES, Graph, and Image formats.

**Last Updated**: 2025-08-19 | **Version**: 2.0.0-production

## 🎉 Current Implementation Status: **PRODUCTION READY**

**All 9 Input-Output Combinations Implemented:**
- ✅ Scaffold(SMILES/Graph/Image) + Text → SMILES  
- ✅ Scaffold(SMILES/Graph/Image) + Text → Graph
- ✅ Scaffold(SMILES/Graph/Image) + Text → Image

**Ready for Training and Deployment!**

## 🎯 Project Overview

Text2Mol implements a sophisticated multi-modal molecular generation system with unified encoder-decoder architecture and MolT5 integration. The system supports scaffold-guided generation across all modalities with progressive implementation strategy.

### Key Innovations
- **Unified Architecture**: Single End2EndMolecularGenerator handles all 9 combinations
- **Progressive Implementation**: Phased approach for rapid deployment ✅ **COMPLETED**
- **Direct Decoders**: Graph and Image decoders alongside SMILES generation
- **Comprehensive Evaluation**: 9 metrics including validity, novelty, and fingerprint similarities

## 🌟 Features

✅ **Complete 9 I/O Combinations**: All implemented and tested  
✅ **Advanced Encoders**: MolT5-Large, GIN, Swin Transformer, BERT/SciBERT  
✅ **Modal Fusion**: Cross-attention + gated fusion mechanisms  
✅ **Scaffold-Guided**: Molecular core structure preservation  
✅ **Production Ready**: 596M parameters, optimized for deployment  
✅ **Comprehensive Testing**: Full test suite with validation
✅ **Direct Decoders**: Graph and Image decoders for direct generation

## 📊 Evaluation Metrics

Our system implements 9 state-of-the-art evaluation metrics:

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
| **FCD** | Fréchet ChemNet Distance | <5.0 |

## 🚀 Quick Start Guide

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

### Immediate Training
```bash
# Start unified multi-modal training
python train_unified_multimodal.py --batch-size 4 --epochs 10

# Test all 9 I/O combinations
python test_9_combinations.py --device cuda

# Quick functionality check
python quick_test.py
```

### Key Implementation Files
- **`train_unified_multimodal.py`** - Complete multi-modal training pipeline
- **`test_9_combinations.py`** - Comprehensive testing of all 9 combinations
- **`scaffold_mol_gen/models/end2end_model.py`** - Unified multi-modal generator
- **`scaffold_mol_gen/models/graph_decoder.py`** - Molecular graph decoder
- **`scaffold_mol_gen/models/image_decoder.py`** - Molecular image decoder

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Layer                               │
│  Scaffold (SMILES/Image/Graph) + Text Description          │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│                 Encoder Layer                                │
│  [BartSMILES] [SciBERT] [GIN] [Swin Transformer]          │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│              Modal Fusion Layer                             │
│         Cross-attention + Feature Alignment                 │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│               Core Model                                     │
│            MolT5-large + Adapters                          │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│              Decoder Layer                                   │
│     [SMILES] [Graph] [Image] Decoders                      │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
scaffold-mol-generation/
├── scaffold_mol_gen/           # Core library
│   ├── models/                 # Neural network models
│   │   ├── encoders/          # Multi-modal encoders
│   │   ├── fusion_simplified.py # Modal fusion layer
│   │   ├── molt5_adapter.py   # MolT5 integration
│   │   ├── graph_decoder.py   # Graph decoder
│   │   ├── image_decoder.py   # Image decoder
│   │   └── end2end_model.py   # Complete pipeline
│   ├── data/                  # Data processing
│   ├── training/              # Training utilities
│   ├── evaluation/            # Metrics and evaluation
│   └── utils/                 # Helper functions
├── train_unified_multimodal.py # Unified training
├── test_9_combinations.py     # Full 9-combo testing
├── quick_test.py              # Basic functionality test
└── docs/                      # Documentation
```

## 📊 Supported Input-Output Combinations (9 Total)

### Complete Multi-Modal I/O Matrix

| Scaffold Input | Text Input | Output | Status | Implementation |
|----------------|------------|--------|--------|----------------|
| SMILES | ✓ | SMILES | ✅ **Implemented** | End2End Model |
| SMILES | ✓ | Graph | ✅ **Implemented** | Direct Graph Decoder |
| SMILES | ✓ | Image | ✅ **Implemented** | Direct Image Decoder |
| Graph | ✓ | SMILES | ✅ **Implemented** | End2End Model |
| Graph | ✓ | Graph | ✅ **Implemented** | Direct Graph Decoder |
| Graph | ✓ | Image | ✅ **Implemented** | Direct Image Decoder |
| Image | ✓ | SMILES | ✅ **Implemented** | End2End Model |
| Image | ✓ | Graph | ✅ **Implemented** | Direct Graph Decoder |
| Image | ✓ | Image | ✅ **Implemented** | Direct Image Decoder |

## 📊 Dataset

The system uses the ChEBI-20 dataset with multi-modal annotations:
- **Size**: 33,010 molecules
- **Modalities**: SMILES, molecular graphs, 2D images, text descriptions
- **Splits**: Train (80%), Validation (10%), Test (10%)

## 🚀 Progressive Implementation Strategy ✅ **COMPLETED**

### Phase 1: Core SMILES Generation ✅
**Goal**: Fix and optimize text-to-SMILES generation

- ✅ Multi-modal encoders implementation
- ✅ Fusion layer architecture
- ✅ MolT5-Large integration
- ✅ SMILES validity constraints

### Phase 2: Graph Decoder Implementation ✅
**Goal**: Enable molecular graph generation

- ✅ Lightweight graph decoder using reverse GNN
- ✅ Adjacency matrix prediction head
- ✅ Graph validity checks
- ✅ Graph reconstruction training

### Phase 3: Image Decoder Integration ✅
**Goal**: Generate 2D molecular structure images

- ✅ RDKit rendering as decoder
- ✅ Lightweight VAE for direct generation
- ✅ Image generation and validation

### Phase 4: Unified Training Framework ✅
**Goal**: Joint training of all modalities

- ✅ UnifiedMultiModalGenerator implementation
- ✅ Progressive training strategy
- ✅ All 9 I/O combinations working
- ✅ Comprehensive test suite

## 📚 Documentation

- **[CLAUDE.md](CLAUDE.md)** - Complete implementation guide and current status
- **Implementation Files**: All key components documented inline
- **Test Suite**: Comprehensive testing with `test_9_combinations.py`

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