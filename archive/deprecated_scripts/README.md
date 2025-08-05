# 归档脚本说明

这个目录包含了已被更新版本替代的脚本。

## 训练脚本
- **train.py** - 基础训练脚本，已被train_fast_stable.py替代
- **train_stable.py** - 稳定版训练，已被train_fast_stable.py替代
- **train_fast.py** - 快速训练版本，已被train_fast_stable.py替代
- **train_dual_gpu.py** - 双GPU训练脚本，特定场景使用

## 推荐使用
- 单GPU训练：使用 `train_fast_stable.py`
- 多模态训练：使用 `train_multimodal.py`

归档日期：2025-08-05