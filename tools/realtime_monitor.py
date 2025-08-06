#!/usr/bin/env python3
"""
实时训练监控
"""

import subprocess
import time
import os
from datetime import datetime

def monitor():
    print("📊 实时训练监控")
    print("按Ctrl+C退出\n")
    
    start_time = time.time()
    
    while True:
        # 清屏
        os.system('clear')
        
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        
        print(f"📊 训练监控 | 运行时间: {hours}h {minutes}m")
        print("=" * 60)
        
        # GPU状态
        print("\n🎮 GPU状态:")
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", 
             "--format=csv,noheader"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                parts = line.split(', ')
                if len(parts) >= 4:
                    idx, util, mem_used, mem_total = parts
                    print(f"  GPU {idx}: 使用率 {util} | 显存 {mem_used}/{mem_total}")
        
        # 进程状态
        print("\n🔄 训练进程:")
        result = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'train_multimodal' in line and 'grep' not in line:
                    parts = line.split()
                    if len(parts) > 10:
                        pid = parts[1]
                        cpu = parts[2]
                        mem = parts[3]
                        # 提取模态名称
                        if 'smiles' in line:
                            modality = 'SMILES'
                        elif 'graph' in line:
                            modality = 'Graph'
                        elif 'image' in line:
                            modality = 'Image'
                        else:
                            modality = 'Unknown'
                        print(f"  {modality}: PID {pid} | CPU {cpu}% | MEM {mem}%")
        
        # 日志最新行
        print("\n📝 最新日志:")
        for log_file, name in [
            ("logs/smiles_train.log", "SMILES"),
            ("logs/graph_train.log", "Graph")
        ]:
            if os.path.exists(log_file):
                result = subprocess.run(
                    ["tail", "-1", log_file],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    last_line = result.stdout.strip()
                    if last_line:
                        # 截取最后80个字符
                        if len(last_line) > 80:
                            last_line = "..." + last_line[-77:]
                        print(f"  {name}: {last_line}")
        
        # 磁盘状态
        print("\n💾 磁盘状态:")
        result = subprocess.run(
            ["df", "-h", "/root/autodl-tmp"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                parts = lines[1].split()
                if len(parts) >= 5:
                    used = parts[2]
                    total = parts[1]
                    percent = parts[4]
                    available = parts[3]
                    print(f"  使用: {used}/{total} ({percent}) | 可用: {available}")
        
        time.sleep(5)  # 每5秒更新

if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("\n\n监控已停止")