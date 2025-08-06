# 🖥️ Terminal-Friendly Quick Reference

适合终端快速查看的精简架构图和命令参考。

## ⚡ 快速架构概览

```
Text2Mol Multi-Modal Molecular Generation Pipeline
==================================================

INPUT          ENCODING         FUSION          GENERATION        OUTPUT
-----          --------         ------          ----------        ------

SMILES ────► MolT5-Large ────┐
Graph  ────► 5-layer GIN ────┤
Image  ────► Swin Trans  ────┼──► Cross-Attention ──► MolT5 Adapter ──► SMILES ──► Graph
Text   ────► BERT/SciBERT ───┘      + Gating          768→1024       Image

                768-dim Features    768-dim Fused      Generation
                                                      (Beam Search)
```

## 🔄 核心工作流程

```
1. TRAINING FLOW:
   Data Loading → Multi-Modal Encoding → Fusion → MolT5 Generation → Loss Calculation
                                                                    → Parameter Update
                                                                    → Validation Check

2. INFERENCE FLOW:
   Input Processing → Modality Detection → Encoding → Fusion → Generation → Output Decoding

3. EVALUATION FLOW:
   Model Loading → 9 Combinations Testing → Metrics Calculation → Results Aggregation
```

## 📊 支持的模态组合

```
┌─────────────┬─────────────┬─────────────┬─────────────┐
│   INPUT     │ → SMILES    │ → GRAPH     │ → IMAGE     │
├─────────────┼─────────────┼─────────────┼─────────────┤
│ SMILES+Text │     ✅      │     ✅      │     ✅      │
│ Graph+Text  │     ✅      │     ✅      │     ✅      │
│ Image+Text  │     ✅      │     ✅      │     ✅      │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

## 🎯 常用终端命令

```bash
# 项目结构查看
tree -L 3 scaffold_mol_gen/

# 快速测试
python test_e2e_simple.py

# 完整评估
python real_model_evaluation.py

# 查看配置
ls configs/*.yaml

# 检查模型
ls -la models/

# 查看评估结果
ls evaluation_results/

# Git状态检查
git status --short
```

## 📈 关键性能指标

```
Parameters:  596.52M (59.08M trainable)
GPU Memory:  ~8GB (batch_size=2) 
Speed:       ~0.5s/sample
Modalities:  9 combinations (3×3)
Accuracy:    See evaluation_results/
```

## 🔧 故障排除

```
Common Issues:
- CUDA Memory: Reduce batch_size in configs/
- Import Error: Check requirements.txt
- Model Missing: Verify model symlinks
- Data Error: Re-run preprocessing scripts
```

## 📝 文件快速定位

```
Core Code:        scaffold_mol_gen/models/
Configurations:   configs/
Documentation:    docs/
Test Scripts:     tests/
Results:          evaluation_results/
Archive:          archive/
```

---
*Use: cat docs/TERMINAL_VIEW.md for quick reference*