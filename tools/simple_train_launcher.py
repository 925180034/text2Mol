#!/usr/bin/env python3
"""
最简单的训练启动器 - 直接使用命令行
"""

import os
import time

# 确保目录存在
os.makedirs("logs", exist_ok=True)
os.makedirs("/root/autodl-tmp/text2Mol-outputs/fast_training/smiles", exist_ok=True)
os.makedirs("/root/autodl-tmp/text2Mol-outputs/fast_training/graph", exist_ok=True)
os.makedirs("/root/autodl-tmp/text2Mol-outputs/fast_training/image", exist_ok=True)

print("🚀 快速训练启动器")
print("=" * 60)

# 检查GPU状态
print("\n📊 GPU状态:")
os.system("nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv")

print("\n开始启动训练任务...")

# SMILES训练 - GPU 0
print("\n1. 启动SMILES训练 (GPU 0, batch=32)...")
cmd1 = "CUDA_VISIBLE_DEVICES=0 python train_multimodal.py --scaffold-modality smiles --batch-size 32 --epochs 1 --lr 3e-5 --output-dir /root/autodl-tmp/text2Mol-outputs/fast_training/smiles > logs/smiles_train.log 2>&1 &"
os.system(cmd1)
print("   ✅ SMILES训练已在后台启动")

time.sleep(5)

# Graph训练 - GPU 1  
print("\n2. 启动Graph训练 (GPU 1, batch=16)...")
cmd2 = "CUDA_VISIBLE_DEVICES=1 python train_multimodal.py --scaffold-modality graph --batch-size 16 --epochs 1 --lr 2e-5 --output-dir /root/autodl-tmp/text2Mol-outputs/fast_training/graph > logs/graph_train.log 2>&1 &"
os.system(cmd2)
print("   ✅ Graph训练已在后台启动")

print("\n" + "=" * 60)
print("✅ 所有训练任务已启动!")
print("\n监控命令:")
print("  查看SMILES日志: tail -f logs/smiles_train.log")
print("  查看Graph日志: tail -f logs/graph_train.log")
print("  查看GPU使用: nvidia-smi -l 1")
print("  查看进程: ps aux | grep train_multimodal")

print("\n预计训练时间:")
print("  1个epoch: 约30-45分钟")
print("  如果效果不错，可以继续训练更多epochs")

print("\n💡 提示:")
print("  - 如果内存不足，减小batch size")
print("  - 训练会自动保存checkpoint到output-dir")
print("  - 可以使用--resume参数从checkpoint恢复训练")