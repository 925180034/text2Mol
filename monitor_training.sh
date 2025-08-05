#!/bin/bash

echo "=== 训练监控脚本 ==="
echo

# 检查训练进程
echo "训练进程状态："
ps aux | grep train_fast_stable.py | grep -v grep | head -1

echo
echo "GPU使用情况："
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | awk -F', ' '{printf "GPU利用率: %s%%\n内存使用: %s/%s MB\n温度: %s°C\n", $1, $2, $3, $4}'

echo
echo "最新训练进度："
tail -10 logs/safe_fast_training.log | grep -E "Batch|Loss|Epoch|completed|Estimated" | tail -5

echo
echo "损失曲线："
grep "Avg:" logs/safe_fast_training.log | tail -10 | awk '{print $9}' | awk -F, '{print "平均损失: " $1}'

echo
echo "预计完成时间："
tail -50 logs/safe_fast_training.log | grep "Estimated completion" | tail -1

echo
echo "持续监控命令："
echo "  tail -f logs/safe_fast_training.log"