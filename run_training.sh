#!/bin/bash
# 统一的训练运行脚本

echo "=== 分子生成模型训练 ==="
echo "使用配置: configs/default_config.yaml"
echo ""

# 检查是否有正在运行的训练
if pgrep -f "train_multimodal.py" > /dev/null; then
    echo "⚠️ 警告: 已有训练进程在运行!"
    exit 1
fi

# 创建必要的目录
mkdir -p logs checkpoints outputs

# 运行训练
echo "开始训练..."
python train_multimodal.py --config configs/default_config.yaml 2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log

echo "训练完成！"