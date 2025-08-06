# Text2Mol ASCII 架构图和流程图

适用于终端查看的纯文本图表，无需任何渲染工具。

## 🏗️ 系统架构图 (ASCII)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            Text2Mol 多模态分子生成系统                            │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                                INPUT LAYER                                     │
├─────────────────┬─────────────────┬─────────────────┬─────────────────────────┤
│   SMILES        │     Graph       │      Image      │      Text              │
│   Scaffold      │    Scaffold     │     Scaffold    │    Description         │
│                 │                 │                 │                        │
│ "CCO", "C=C"    │  PyTorch        │   299×299 RGB   │ "ethanol molecule"     │
│                 │  Geometric      │    Images       │                        │
└─────────────────┴─────────────────┴─────────────────┴─────────────────────────┘
         │                 │                 │                 │
         ▼                 ▼                 ▼                 ▼
┌─────────────────┬─────────────────┬─────────────────┬─────────────────────────┐
│                                ENCODING LAYER                                │
├─────────────────┼─────────────────┼─────────────────┼─────────────────────────┤
│  SMILES Encoder │  Graph Encoder  │  Image Encoder  │    Text Encoder        │
│                 │                 │                 │                        │
│   MolT5-Large   │   5-layer GIN   │ Swin Transform  │   BERT/SciBERT         │
│     (3GB)       │    Network      │      Base       │                        │
│                 │                 │                 │                        │
│ Input: Tokens   │ Input: Nodes+   │ Input: Pixels   │ Input: Tokens          │
│ Output: 768-dim │       Edges     │ Output: 768-dim │ Output: 768-dim        │
│                 │ Output: 768-dim │                 │                        │
└─────────────────┴─────────────────┴─────────────────┴─────────────────────────┘
         │                 │                 │                 │
         └─────────────────┼─────────────────┼─────────────────┘
                           ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            FUSION LAYER                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                        Multi-Modal Fusion                                      │
│                                                                                │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐    │
│  │ Cross-Attention │    │ Gated Fusion    │    │   Combined Strategy     │    │
│  │                 │    │                 │    │                         │    │
│  │ Multi-Head      │ +  │ Learnable Gates │ =  │ Optimal Feature Fusion  │    │
│  │ Attention       │    │ Dynamic Weights │    │                         │    │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────┘    │
│                                                                                │
│                        Output: 768-dim Fused Features                         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          GENERATION LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                           MolT5 Adapter                                        │
│                                                                                │
│  768-dim ──► Linear+LayerNorm+GELU ──► 1024-dim ──► Sequence Expansion        │
│              Dimension Adaptation                    Position Encoding         │
│                                                                                │
│                         MolT5 Generator                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                    MolT5-Large Backbone                                 │  │
│  │                                                                         │  │
│  │  Beam Search (3-5 beams) + Temperature (0.8) + Max Length (128)        │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │    Generated SMILES     │
                         │                         │
                         │     "C1=CC=CC=C1"       │
                         └─────────────────────────┘
                                      │
                         ┌────────────┼────────────┐
                         ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           OUTPUT LAYER                                         │
├─────────────────┬─────────────────────────────┬─────────────────────────────────┤
│   SMILES Output │      Graph Output           │       Image Output              │
│                 │                             │                                 │
│  Direct Return  │  SMILES→Graph Decoder       │   SMILES→Image Decoder          │
│                 │                             │                                 │
│ "C1=CC=CC=C1"   │  RDKit + PyTorch Geometric  │   RDKit + PIL                   │
│                 │                             │                                 │
│                 │  Output: Data(x, edge_idx)  │   Output: 299×299 RGB Array     │
└─────────────────┴─────────────────────────────┴─────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                            SUPPORTED COMBINATIONS                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  Input Modalities           →  Output Modalities                              │
│                                                                                │
│  ┌─────────────────┐           ┌─────────────────┐                             │
│  │ SMILES + Text   │────────►  │ SMILES ✅       │                             │
│  └─────────────────┘     ┌────►│ Graph  ✅       │                             │
│                         │     │ Image  ✅       │                             │
│  ┌─────────────────┐    │      └─────────────────┘                             │
│  │ Graph + Text    │────┤                                                      │
│  └─────────────────┘    │      ┌─────────────────┐                             │
│                         │     │                  │                             │
│  ┌─────────────────┐    │     │  7 Combinations  │                             │
│  │ Image + Text    │────┘     │    Supported     │                             │
│  └─────────────────┘          │                  │                             │
│                               └─────────────────┘                             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 🔄 主要工作流程图 (ASCII)

### 1. 模型推理流程

```
START
  │
  ▼
┌─────────────────────────┐
│    Input Processing     │
│                         │
│ Scaffold + Text Input   │
│ Format Validation       │
└─────────────────────────┘
  │
  ▼
┌─────────────────────────┐
│   Modality Detection    │
│                         │
│ ┌─────┐ ┌─────┐ ┌─────┐ │
│ │SMILE││Graph││Image│ │
│ │  S  ││     ││     │ │
│ └─────┘ └─────┘ └─────┘ │
└─────────────────────────┘
  │
  ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│   Scaffold Encoding    │────►│    Text Encoding        │
│                         │     │                         │
│ • MolT5 (SMILES)        │     │ • BERT/SciBERT          │
│ • GIN (Graph)           │     │ • 768-dim features      │
│ • Swin (Image)          │     │                         │
│ • 768-dim output        │     │                         │
└─────────────────────────┘     └─────────────────────────┘
  │                                         │
  └─────────────────┬─────────────────────────┘
                    ▼
┌─────────────────────────────────────────────────┐
│              Modal Fusion                       │
│                                                 │
│  Scaffold Features + Text Features              │
│            │                                    │
│            ▼                                    │
│  ┌─────────────────┐ ┌─────────────────┐       │
│  │ Cross-Attention │+│ Gated Fusion    │       │
│  └─────────────────┘ └─────────────────┘       │
│            │                                    │
│            ▼                                    │
│     768-dim Fused Features                      │
└─────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────┐
│              MolT5 Generation                   │
│                                                 │
│  768-dim ──► Adapter ──► 1024-dim ──► Seq      │
│                    │                            │
│                    ▼                            │
│  ┌───────────────────────────────────────────┐  │
│  │           Beam Search                     │  │
│  │                                           │  │
│  │  Beam 1: "CCO"      Score: 0.95          │  │
│  │  Beam 2: "C(C)O"    Score: 0.87          │  │
│  │  Beam 3: "CC[OH]"   Score: 0.82          │  │
│  └───────────────────────────────────────────┘  │
│                    │                            │
│                    ▼                            │
│            Selected: "CCO"                      │
└─────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────┐
│            Output Decoding                      │
│                                                 │
│              Generated SMILES                   │
│                     │                           │
│      ┌──────────────┼──────────────┐           │
│      ▼              ▼              ▼           │
│ ┌─────────┐  ┌─────────────┐  ┌──────────────┐ │
│ │ SMILES  │  │ Graph       │  │ Image        │ │
│ │ Direct  │  │ Decoder     │  │ Decoder      │ │
│ │ Output  │  │             │  │              │ │
│ │         │  │ RDKit +     │  │ RDKit +      │ │
│ │ "CCO"   │  │ PyG         │  │ PIL          │ │
│ └─────────┘  └─────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────┐
│   Output Validation     │
│                         │
│ • Molecular validity    │
│ • Chemical plausibility │
│ • Format consistency    │
└─────────────────────────┘
  │
  ▼
END
```

### 2. 训练流程 (简化版)

```
┌─────────────────────────┐
│   Training Start        │
│                         │
│ • Config Loading        │
│ • Model Initialization  │
└─────────────────────────┘
  │
  ▼
┌─────────────────────────┐
│   Data Loading          │
│                         │
│ • MultimodalDataset     │
│ • Batch Processing      │
│ • Data Augmentation     │
└─────────────────────────┘
  │
  ▼
╔═════════════════════════╗  ◄─────── Training Loop
║     Forward Pass        ║
║                         ║
║ Batch Data              ║
║   │                     ║
║   ▼                     ║
║ Multi-Modal Encoding    ║
║   │                     ║
║   ▼                     ║
║ Modal Fusion            ║
║   │                     ║
║   ▼                     ║
║ MolT5 Generation        ║
║   │                     ║
║   ▼                     ║
║ Loss Calculation        ║
║ (CrossEntropyLoss)      ║
╚═════════════════════════╝
  │
  ▼
┌─────────────────────────┐
│   Backward Pass         │
│                         │
│ • Gradient Computation  │
│ • Parameter Update      │
│ • Learning Rate Adjust  │
└─────────────────────────┘
  │
  ▼
┌─────────────────────────┐      YES
│   Validation Check?     │──────────┐
│                         │          │
│ Every N Steps           │          ▼
└─────────────────────────┘    ┌─────────────────────────┐
  │ NO                         │   Validation Run        │
  │                            │                         │
  │                            │ • BLEU Score            │
  │                            │ • Validity Rate         │
  │                            │ • Similarity Metrics    │
  │                            └─────────────────────────┘
  │                                      │
  │                                      ▼
  │                            ┌─────────────────────────┐
  │                            │   Model Checkpointing   │
  │                            │                         │
  │                            │ • Save Best Model       │
  │                            │ • Early Stop Check      │
  │                            └─────────────────────────┘
  │                                      │
  │                                      │
  └──────────────────────────────────────┘
  │
  ▼
┌─────────────────────────┐
│   Training Complete     │
│                         │
│ • Final Model Save      │
│ • Training Summary      │
└─────────────────────────┘
```

### 3. 评估流程 (九种模态组合)

```
                           EVALUATION START
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │    Load Trained Models      │
                    │                             │
                    │  • SMILES Model             │
                    │  • Graph Model              │
                    │  • Image Model              │
                    └─────────────────────────────┘
                                  │
                                  ▼
      ┌───────────────────────────────────────────────────────────────┐
      │                   9 Modality Combinations                    │
      │                                                               │
      │  Input Modalities    →    Output Modalities                  │
      │                                                               │
      │  ┌─────────────────┐      ┌─────────────────┐                │
      │  │ SMILES + Text   │────► │ SMILES          │ ✅ Combo 1     │
      │  │                 │ ┌──► │ Graph           │ ✅ Combo 2     │
      │  │                 │ │ ┌► │ Image           │ ✅ Combo 3     │
      │  └─────────────────┘ │ │  └─────────────────┘                │
      │                      │ │                                     │
      │  ┌─────────────────┐ │ │  ┌─────────────────┐                │
      │  │ Graph + Text    │─┼─┼► │ SMILES          │ ✅ Combo 4     │
      │  │                 │ │ ├► │ Graph           │ ✅ Combo 5     │
      │  │                 │ │ │► │ Image           │ ✅ Combo 6     │
      │  └─────────────────┘ │ │  └─────────────────┘                │
      │                      │ │                                     │
      │  ┌─────────────────┐ │ │  ┌─────────────────┐                │
      │  │ Image + Text    │─┘ │  │ SMILES          │ ✅ Combo 7     │
      │  │                 │───┼► │ Graph           │ ✅ Combo 8     │
      │  │                 │───┘► │ Image           │ ✅ Combo 9     │
      │  └─────────────────┘      └─────────────────┘                │
      └───────────────────────────────────────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
┌───────────────┐    ┌───────────────────┐    ┌─────────────────┐
│  Combo 1-3    │    │    Combo 4-6      │    │   Combo 7-9     │
│  Processing   │    │   Processing       │    │   Processing    │
│               │    │                    │    │                 │
│ SMILES Input  │    │  Graph Input       │    │  Image Input    │
│ 100 samples   │    │  100 samples       │    │  100 samples    │
└───────────────┘    └───────────────────┘    └─────────────────┘
        │                         │                         │
        └─────────────────────────┼─────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      EVALUATION METRICS                            │
│                                                                     │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────┐   │
│  │  Molecular      │ │   Sequence      │ │   Chemical          │   │
│  │  Quality        │ │   Similarity    │ │   Similarity        │   │
│  │                 │ │                 │ │                     │   │
│  │ • Validity      │ │ • BLEU          │ │ • MACCS Tanimoto    │   │
│  │ • Uniqueness    │ │ • Exact Match   │ │ • Morgan Tanimoto   │   │
│  │ • Novelty       │ │ • Levenshtein   │ │ • RDKit Tanimoto    │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────────┘   │
│                                  │                                 │
│                                  ▼                                 │
│                    ┌─────────────────────────────┐                 │
│                    │     Distribution            │                 │
│                    │     Distance                │                 │
│                    │                             │                 │
│                    │ • FCD (Fréchet ChemNet)     │                 │
│                    └─────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │    Results Aggregation      │
                    │                             │
                    │ • JSON Export               │
                    │ • Markdown Report           │
                    │ • Visualization             │
                    │ • Performance Summary       │
                    └─────────────────────────────┘
                                  │
                                  ▼
                           EVALUATION END
```

## 📊 性能指标 (ASCII表格)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        System Performance                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Component           │  Specification                               │
│  ─────────────────────┼─────────────────────────────────────        │
│  Total Parameters    │  596.52M (59.08M trainable, 537.44M frozen) │
│  GPU Memory          │  ~8GB (batch_size=2)                        │
│  Inference Speed     │  ~0.5s/sample                               │
│  Training Time       │  ~12h (single 32GB GPU)                     │
│  Model Size          │  ~3GB (MolT5-Large backbone)                │
│                                                                     │
│  ─────────────────────┼─────────────────────────────────────        │
│  Supported Modalities                                               │
│  ─────────────────────┼─────────────────────────────────────        │
│  Input Combinations  │  3 (SMILES, Graph, Image) + Text            │
│  Output Combinations │  3 (SMILES, Graph, Image)                   │
│  Total Combinations  │  9 (3×3)                                     │
│                                                                     │
│  ─────────────────────┼─────────────────────────────────────        │
│  Data Specifications                                                │
│  ─────────────────────┼─────────────────────────────────────        │
│  Feature Dimension   │  768-dim (unified across all modalities)    │
│  Image Resolution    │  299×299×3 (RGB)                            │
│  Max Sequence Length │  128 tokens                                  │
│  Dataset Size        │  33,010 ChEBI-20 records                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 🎯 快速命令参考

### 终端查看命令
```bash
# 查看架构图
cat docs/ASCII_DIAGRAMS.md | grep -A 50 "系统架构图"

# 查看流程图  
cat docs/ASCII_DIAGRAMS.md | grep -A 100 "推理流程"

# 查看性能指标
cat docs/ASCII_DIAGRAMS.md | grep -A 20 "性能指标"
```

### 项目结构概览
```
text2Mol/scaffold-mol-generation/
├── scaffold_mol_gen/           # 核心代码包
│   ├── models/                 # 神经网络模型
│   │   ├── encoders/           # 多模态编码器
│   │   ├── fusion_simplified.py # 模态融合
│   │   ├── molt5_adapter.py    # MolT5适配器
│   │   └── end2end_model.py    # 端到端模型
│   ├── data/                   # 数据处理
│   ├── training/              # 训练相关
│   └── utils/                 # 工具函数
├── configs/                   # 配置文件
├── docs/                      # 文档
├── evaluation_results/        # 评估结果
└── tests/                     # 测试脚本
```

---
*ASCII图表兼容所有终端，无需额外工具*