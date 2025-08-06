# 多模态数据说明

## 您的问题解答

**问题**: "为什么我看不到graph和image的模态的数据，我的这两个模态数据在哪里？"

**答案**: ChEBI-20数据集中**原本就没有**预先生成的graph和image数据。系统采用**动态转换**的方式来支持多模态。

## 数据集现状

### 原始数据结构
```
Datasets/train.csv包含的列:
- CID: 化合物ID
- SMILES: 分子的SMILES表示 ✅ (唯一的结构化数据)
- description: 文本描述 ✅
- polararea, xlogp, inchi, iupacname, SELFIES: 其他化学属性
```

### 缺失的数据
- ❌ **没有Graph列**: 没有预先计算的分子图结构
- ❌ **没有Image列**: 没有预先生成的分子图像
- ❌ **没有预处理的多模态特征**

## 多模态数据生成机制

### 动态转换流程
1. **输入**: 只有SMILES字符串
2. **转换**: 根据`scaffold_modality`参数动态生成
3. **输出**: 三种不同表示的同一个分子

```python
# 训练时指定不同模态
python train_multimodal.py --scaffold-modality smiles  # 原始SMILES
python train_multimodal.py --scaffold-modality graph   # 动态生成图结构
python train_multimodal.py --scaffold-modality image   # 动态生成分子图像
```

### 转换详情

#### SMILES → Graph转换
```python
# 使用MultiModalPreprocessor.smiles_to_graph()
# SMILES: "CCO" → Graph特征:
# - 节点数: 3 (C, C, O)
# - 节点特征: 146维 (原子类型、度数、杂化等)
# - 边数: 4 (C-C, C-O, 双向)
# - 边特征: 8维 (键类型、共轭性等)
```

#### SMILES → Image转换
```python
# 使用MultiModalPreprocessor.smiles_to_image()
# SMILES: "CCO" → Image特征:
# - 尺寸: 224×224×3 (RGB)
# - 格式: 2D分子结构图
# - 数据类型: uint8 [0-255]
```

## 验证结果

通过运行`check_multimodal_data.py`验证：

```
✅ Graph生成成功率: 10/10 (100%)
✅ Image生成成功率: 10/10 (100%)

示例输出:
- 节点数: 55
- 节点特征维度: 146  
- 边数: 110
- 边特征维度: 8
- 图像尺寸: (224, 224, 3)
```

## 为什么采用动态转换？

### 优势
1. **存储效率**: 不需要预存大量图像和图结构数据
2. **灵活性**: 可以随时调整图像尺寸、图特征等参数
3. **一致性**: 保证所有模态来源于同一个SMILES，避免数据不匹配

### 代价
1. **计算开销**: 训练时需要实时转换
2. **内存使用**: 需要临时存储转换后的数据
3. **速度影响**: 相比预处理数据会慢一些

## 如何确认多模态工作？

### 检查方法
```bash
# 1. 验证数据转换功能
python check_multimodal_data.py

# 2. 测试不同模态训练
python test_all_modalities.py

# 3. 运行多模态评估
python demo_multimodal_evaluation.py
```

### 训练不同模态
```bash
# SMILES模态训练
./launch_short_term_training.sh --scaffold-modality smiles

# Graph模态训练（需要更多内存和时间）
./launch_short_term_training.sh --scaffold-modality graph

# Image模态训练（需要最多内存和时间）  
./launch_short_term_training.sh --scaffold-modality image
```

## 总结

您的数据集是**正常的**！Graph和Image数据**不应该**预先存在于CSV文件中，而是应该在训练/推理时动态生成。这是这个多模态分子生成系统的设计特点：

- **单一数据源**: SMILES字符串
- **多种表示**: 动态转换为Graph/Image
- **统一训练**: 相同的模型架构处理不同模态

当您指定不同的`--scaffold-modality`参数时，系统会自动将SMILES转换为对应的模态数据进行训练。