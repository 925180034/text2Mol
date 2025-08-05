#\!/bin/bash
# 统一的训练监控脚本

echo "=== 训练监控脚本 ==="
echo "用法: ./monitor_training.sh [日志文件]"
echo ""

LOG_FILE=${1:-"training.log"}

if [ \! -f "$LOG_FILE" ]; then
    echo "错误: 日志文件 $LOG_FILE 不存在"
    exit 1
fi

echo "监控日志文件: $LOG_FILE"
echo "按 Ctrl+C 退出"
echo ""

# 监控训练进度
tail -f "$LOG_FILE" | grep --line-buffered -E "(Loss:|Epoch:|Valid:|lr:|time:|Saved|Error|Warning)" | while read line; do
    echo "[$(date '+%H:%M:%S')] $line"
done
EOF < /dev/null
