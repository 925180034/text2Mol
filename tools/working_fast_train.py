#!/usr/bin/env python3
"""
实际可用的快速训练脚本
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# 确保目录正确
os.chdir('/root/text2Mol/scaffold-mol-generation')
Path("logs").mkdir(exist_ok=True)

def launch_training_subprocess(modality, gpu_id):
    """使用子进程启动训练，避免导入超时问题"""
    
    print(f"\n🚀 启动 {modality} 训练 (GPU {gpu_id})")
    
    # 创建Python脚本内容
    script_content = f'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '{gpu_id}'

import sys
sys.path.append('/root/text2Mol/scaffold-mol-generation')

print("正在加载模块...")
from train_multimodal import main

# 设置参数
import argparse
args = argparse.Namespace(
    train_data='Datasets/train.csv',
    val_data='Datasets/validation.csv',
    test_data='Datasets/test.csv',
    output_dir='/root/autodl-tmp/text2Mol-outputs/fast_training/{modality}',
    batch_size={{'smiles': 32, 'graph': 16, 'image': 8}[modality]},
    epochs=1,  # 先训练1个epoch
    lr={{'smiles': 3e-5, 'graph': 2e-5, 'image': 1e-5}[modality]},
    scaffold_modality='{modality}',
    resume=None,
    device='cuda',
    config=None,
    modality='{modality}'
)

# 创建输出目录
os.makedirs(args.output_dir, exist_ok=True)

print(f"开始训练 {modality}...")
try:
    # 调用main函数
    import sys
    sys.argv = ['train_multimodal.py']  # 模拟命令行参数
    
    # 直接运行训练
    exec(open('train_multimodal.py').read())
except Exception as e:
    print("训练出错:", e)
    import traceback
    traceback.print_exc()
'''
    
    # 写入临时脚本
    script_path = f"/tmp/train_{modality}_{gpu_id}.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # 启动训练
    log_file = f"logs/train_{modality}_{time.strftime('%H%M%S')}.log"
    
    cmd = [sys.executable, script_path]
    
    print(f"  日志: {log_file}")
    
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT
        )
    
    print(f"  ✅ PID: {process.pid}")
    return process

def simple_direct_train():
    """直接调用训练，最简单的方式"""
    
    print("🚀 直接训练模式")
    print("=" * 60)
    
    # 检查GPU
    os.system("nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv")
    
    print("\n开始训练...")
    
    # 使用最简单的方式 - 直接命令行调用
    commands = [
        # SMILES on GPU 0
        "CUDA_VISIBLE_DEVICES=0 python train_multimodal.py --scaffold-modality smiles --batch-size 32 --epochs 1 --output-dir /root/autodl-tmp/text2Mol-outputs/fast_training/smiles > logs/smiles.log 2>&1 &",
        
        # Graph on GPU 1
        "CUDA_VISIBLE_DEVICES=1 python train_multimodal.py --scaffold-modality graph --batch-size 16 --epochs 1 --output-dir /root/autodl-tmp/text2Mol-outputs/fast_training/graph > logs/graph.log 2>&1 &"
    ]
    
    for cmd in commands:
        print(f"\n执行: {cmd}")
        os.system(cmd)
        time.sleep(5)
    
    print("\n✅ 训练已在后台启动!")
    print("\n查看进度:")
    print("  tail -f logs/smiles.log")
    print("  tail -f logs/graph.log")
    print("\n查看GPU:")
    print("  nvidia-smi -l 1")
    print("\n查看进程:")
    print("  ps aux | grep train_multimodal")

def main():
    print("选择启动方式:")
    print("1. 子进程方式（推荐）")
    print("2. 直接命令行方式")
    
    choice = input("请选择 (1/2，默认1): ").strip() or "1"
    
    if choice == "1":
        # 子进程方式
        processes = {}
        processes['smiles'] = launch_training_subprocess('smiles', 0)
        time.sleep(10)
        processes['graph'] = launch_training_subprocess('graph', 1)
        
        print("\n监控训练...")
        while any(p.poll() is None for p in processes.values()):
            time.sleep(30)
            status = "状态: "
            for name, p in processes.items():
                if p.poll() is None:
                    status += f"{name}:运行中 "
                else:
                    status += f"{name}:完成 "
            print(f"\r{status}", end='', flush=True)
            
    else:
        # 直接命令行方式
        simple_direct_train()

if __name__ == "__main__":
    # 如果有参数，直接使用命令行方式
    if len(sys.argv) > 1 and sys.argv[1] == "--direct":
        simple_direct_train()
    else:
        main()