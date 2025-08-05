# 模型评估指南

## 🚀 快速开始

### 1. 快速测试（验证模型是否正常）
```bash
python quick_test.py
```
这会生成一个简单的测试结果和可视化图片。

### 2. 运行完整评估（推荐）
```bash
./run_evaluation.sh
```
按提示选择：
- 选项1：测试100个样本（约5分钟）
- 选项2：测试500个样本（约20分钟）
- 选项3：测试全部3476个样本（约2小时）

## 📊 评估指标说明

### 所有9个指标都会被计算：

1. **Validity（有效性）**：生成的SMILES是否为有效的分子结构
2. **Uniqueness（唯一性）**：生成分子的去重比例
3. **Novelty（新颖性）**：生成的分子与输入骨架的差异程度
4. **BLEU Score**：生成序列与目标序列的相似度（0-1）
5. **Exact Match（精确匹配）**：完全匹配目标分子的比例
6. **Levenshtein Distance**：编辑距离，越小越好
7. **MACCS Similarity**：基于MACCS指纹的分子相似度
8. **Morgan Similarity**：基于Morgan指纹的分子相似度
9. **RDK Similarity**：基于RDKit指纹的分子相似度

### 额外指标（如果可用）：
10. **FCD（Fréchet ChemNet Distance）**：生成分子与真实分子在化学空间的距离

## 📁 实验结果目录结构

运行评估后，所有结果会保存在 `experiments/` 目录下：

```
experiments/
└── full_evaluation_20250805_123456/
    ├── experiment_report.md          # 完整的实验报告（推荐阅读）
    ├── inference_results.csv         # 详细的推理结果
    ├── metrics/
    │   ├── all_metrics.json        # 所有指标的JSON格式
    │   └── metrics_summary.txt      # 指标摘要（易读格式）
    ├── visualizations/
    │   └── metrics_bar_chart.png    # 指标柱状图
    └── molecular_images/
        ├── example_0.png            # 生成分子示例
        ├── example_1.png
        └── ...
```

## 🔍 查看结果

### 1. 查看实验报告（最全面）
```bash
cat experiments/full_evaluation_*/experiment_report.md
```

### 2. 查看指标摘要
```bash
cat experiments/full_evaluation_*/metrics/metrics_summary.txt
```

### 3. 查看可视化
在文件浏览器中打开：
- 指标图表：`experiments/full_evaluation_*/visualizations/metrics_bar_chart.png`
- 分子示例：`experiments/full_evaluation_*/molecular_images/`

## 💡 高级用法

### 自定义评估
```bash
python complete_evaluation.py \
    --model /root/autodl-tmp/safe_fast_checkpoints/best_model.pt \
    --test Datasets/test.csv \
    --name my_custom_evaluation \
    --num_samples 200 \
    --save_examples
```

### 批量推理（只生成，不评估）
```bash
python batch_inference.py --num_samples 100 --output results
```

### 单个分子生成
```bash
python inference.py \
    --scaffold "c1ccccc1" \
    --text "A molecule with anti-inflammatory properties" \
    --output my_molecule.png
```

## ⚠️ 注意事项

1. 确保模型文件存在：`/root/autodl-tmp/safe_fast_checkpoints/best_model.pt`
2. 首次运行可能需要加载模型，耗时约1分钟
3. 生成大量样本时请确保有足够的存储空间
4. FCD指标可能需要额外的依赖包

## 📈 结果解读

- **Validity > 0.9**：优秀的分子生成能力
- **Uniqueness > 0.9**：良好的多样性
- **分子相似度 > 0.7**：与目标分子结构相似
- **BLEU > 0.5**：序列生成质量较好