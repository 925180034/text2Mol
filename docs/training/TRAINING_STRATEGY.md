# 多模态训练策略指南

## 🎯 训练目标确认

**是的！通过训练您将实现：**
- ✅ 多模态输入：SMILES、Graph、Image三种形式的scaffold
- ✅ 文本+结构融合：text description + scaffold → 分子生成
- ✅ 跨模态理解：模型学会不同模态间的关系
- ✅ 生成能力提升：从77.8%基线提升到更高的有效性

## ⏰ 训练时间规划

### 快速验证（推荐先做）
```bash
# 2-3小时，验证系统可用性
python start_multimodal_training.py
# 选择选项1：快速验证
```

### 完整多模态训练
```bash
# 8-12小时，获得完整多模态能力
python start_multimodal_training.py  
# 选择选项2：标准训练
```

### 训练时间详细估算
| 模态 | 批次大小 | 每轮时间 | 10轮总时间 | 内存使用 |
|------|----------|----------|------------|----------|
| SMILES | 8 | 15-20分钟 | 2.5-3小时 | ~6GB |
| Graph | 4 | 25-35分钟 | 4-6小时 | ~8GB |
| Image | 2 | 35-45分钟 | 6-8小时 | ~10GB |

## 🚀 三种启动方式

### 方式1：智能启动器（推荐）
```bash
python start_multimodal_training.py
```
- 交互式选择训练方案
- 自动调整批次大小
- 实时监控磁盘空间
- 自动生成训练日志

### 方式2：快速启动脚本
```bash
./launch_short_term_training.sh
```
- 使用预配置参数
- 包含磁盘监控
- SMILES模态，10轮训练

### 方式3：手动精确控制
```bash
# 自定义所有参数
python train_multimodal.py \
    --train-data Datasets/train.csv \
    --val-data Datasets/validation.csv \
    --batch-size 4 \
    --epochs 10 \
    --scaffold-modality graph \
    --lr 5e-5 \
    --device cuda
```

## 📊 预期训练效果

### 训练前（当前状态）
- 有效性：77.8%（仅基础MolT5）
- 模态支持：仅SMILES字符串
- 融合能力：无（scaffold和text独立处理）

### 训练后（预期）
- 有效性：85-90%（多模态融合提升）
- 模态支持：SMILES + Graph + Image
- 融合能力：强（学会scaffold-text协同）
- 新颖性：80-85%（生成新的有效分子）
- 相似性：70-80%（保持scaffold约束）

## 🔍 训练监控

### 实时监控
```bash
# 方法1：查看训练日志
tail -f logs/smiles_training_*.log

# 方法2：监控GPU使用
watch -n 1 nvidia-smi

# 方法3：监控磁盘空间
watch -n 30 df -h /root/autodl-tmp
```

### 关键指标观察
- **Loss下降**：每轮应该有明显下降
- **有效性提升**：验证集有效性从77%向85%+提升
- **生成质量**：生成的SMILES越来越合理
- **内存稳定**：GPU内存使用稳定，无OOM

## 💾 磁盘空间管理

### 当前状态
- 总容量：50GB
- 已使用：19GB (38%)
- 可用空间：32GB

### 训练占用估算
- **模型checkpoint**：每个~2GB
- **日志文件**：每个~100MB
- **临时文件**：~1GB
- **安全余量**：至少保留10GB

### 自动清理机制
系统会自动：
- 删除旧的checkpoint（保留最好的3个）
- 压缩日志文件
- 监控磁盘使用率
- 超过90%时发出警告

## 🎉 训练完成后

### 立即可用功能
```bash
# 1. 基础评估
python final_fixed_evaluation.py --num_samples 100

# 2. 多模态评估
python demo_multimodal_evaluation.py

# 3. 不同模态测试
python test_all_modalities.py
```

### 预期改进
1. **生成质量提升**：更高的分子有效性
2. **模态理解**：可以处理图像和图结构输入
3. **融合能力**：scaffold和文本的协同效果
4. **泛化能力**：对新的分子描述生成合理结果

## 🚨 常见问题预防

### 内存不足
- 减小batch_size：8→4或4→2
- 启用梯度累积：--gradient-accumulation-steps 2

### 磁盘空间不足
- 系统会自动清理
- 手动删除旧实验：rm -rf experiments/old_*

### 训练中断
- 自动保存checkpoint
- 可以从断点继续：--resume path/to/checkpoint

## 💡 建议的执行顺序

1. **立即开始快速验证**（2-3小时）
2. **检查结果质量**
3. **如果满意，继续完整训练**（8-12小时）
4. **评估多模态效果**
5. **调优和部署**

现在就可以开始训练了！建议使用智能启动器获得最佳体验。