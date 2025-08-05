#!/bin/bash
# 统一的评估运行脚本

echo "=== 分子生成模型评估 ==="
echo ""

# 选择评估方式
echo "请选择评估方式:"
echo "1. 快速评估 (10个样本)"
echo "2. 标准评估 (50个样本)"
echo "3. 多模态评估 (30个样本)"
echo "4. 完整评估 (所有样本)"
read -p "请输入选择 (1-4): " choice

case $choice in
    1)
        echo "运行快速评估..."
        python final_fixed_evaluation.py --num_samples 10
        ;;
    2)
        echo "运行标准评估..."
        python final_fixed_evaluation.py --num_samples 50
        ;;
    3)
        echo "运行多模态评估..."
        python demo_multimodal_evaluation.py --num_samples 30
        ;;
    4)
        echo "运行完整评估..."
        python run_multimodal_evaluation.py
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac

echo "评估完成！结果保存在 experiments/ 目录"