# 📊 数据盘清理计划

## 当前状态
- **总空间**: 100GB
- **已使用**: 11GB (11%)
- **可用**: 89GB
- **目标**: 保持在50GB以下

## 占用分析

| 目录 | 大小 | 说明 | 处理方案 |
|------|------|------|----------|
| text2Mol-outputs/molt5_base_20250809_093009 | 5.1GB | 失败的训练（0%有效率） | 🗑️ 删除 |
| text2Mol-outputs/optimized_20250809_105726 | 3.4GB | 当前使用的模型 | ✅ 保留 |
| text2Mol-outputs/optimized_20250809_103* | <1MB | 空目录 | 🗑️ 删除 |
| text2Mol-outputs/optimized_20250809_104* | <1MB | 空目录 | 🗑️ 删除 |
| text2Mol-models/molt5-base | 1.5GB | 基础模型 | ✅ 保留 |
| text2Mol-models/bert-base-uncased | 200MB | BERT模型 | ✅ 保留 |
| text2Mol-models/scibert_scivocab_uncased | 100MB | SciBERT模型 | ✅ 保留 |
| pretrained_models | 337MB | 预训练模型 | ✅ 保留 |
| 实验结果目录 | <50MB | 各种测试结果 | 🗑️ 清理旧的 |

## 清理后预期
- **释放空间**: ~5.2GB
- **剩余使用**: ~5.8GB
- **安全余量**: 94GB可用空间