#!/usr/bin/env python3
"""
Real-time training monitor for Stage 1 alignment training
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime
import argparse

def get_gpu_info():
    """Get GPU memory usage"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', 
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            return {
                'gpu_util': f"{parts[0]}%",
                'memory_used': f"{float(parts[1])/1024:.1f}GB",
                'memory_total': f"{float(parts[2])/1024:.1f}GB",
                'temperature': f"{parts[3]}°C"
            }
    except:
        pass
    return None

def get_disk_usage():
    """Get disk usage for output directory"""
    try:
        result = subprocess.run(
            ['du', '-sh', '/root/autodl-tmp/text2Mol-stage1'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            size = result.stdout.split()[0]
            return size
    except:
        pass
    return "N/A"

def tail_log(log_file, n=10):
    """Get last n lines of log file"""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            return lines[-n:]
    except:
        return []

def monitor(output_dir='/root/autodl-tmp/text2Mol-stage1', refresh_interval=5):
    """Monitor training progress"""
    output_dir = Path(output_dir)
    
    print("=" * 60)
    print("📊 STAGE 1 TRAINING MONITOR")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Refresh interval: {refresh_interval}s")
    print("Press Ctrl+C to exit\n")
    
    try:
        while True:
            # Clear screen
            os.system('clear')
            
            # Header
            print("=" * 60)
            print(f"📊 TRAINING MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)
            
            # GPU Status
            gpu_info = get_gpu_info()
            if gpu_info:
                print("\n🖥️ GPU Status:")
                print(f"  Utilization: {gpu_info['gpu_util']}")
                print(f"  Memory: {gpu_info['memory_used']} / {gpu_info['memory_total']}")
                print(f"  Temperature: {gpu_info['temperature']}")
            
            # Disk Usage
            print(f"\n💾 Model Storage:")
            print(f"  Size: {get_disk_usage()}")
            
            # Find latest log file
            log_files = list(output_dir.glob('training_*.log'))
            if log_files:
                latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
                
                print(f"\n📝 Latest Log: {latest_log.name}")
                print("-" * 60)
                
                # Show last few lines of log
                last_lines = tail_log(latest_log, n=15)
                for line in last_lines:
                    print(line.rstrip())
            else:
                print("\n⏳ Waiting for training to start...")
            
            # Check for checkpoints
            checkpoints = list(output_dir.glob('*.pt'))
            if checkpoints:
                print("\n📦 Saved Models:")
                for ckpt in sorted(checkpoints):
                    size_mb = ckpt.stat().st_size / (1024 * 1024)
                    mtime = datetime.fromtimestamp(ckpt.stat().st_mtime)
                    print(f"  {ckpt.name}: {size_mb:.1f}MB (saved {mtime.strftime('%H:%M:%S')})")
            
            # Check if training is still running
            try:
                result = subprocess.run(['pgrep', '-f', 'train_stage1_optimized.py'], 
                                      capture_output=True)
                if result.returncode == 0:
                    print("\n✅ Training is running (PID: {})".format(
                        result.stdout.decode().strip().split('\n')[0]))
                else:
                    print("\n⚠️ Training process not found")
            except:
                pass
            
            print("\n" + "=" * 60)
            print("Press Ctrl+C to exit monitor")
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitor stopped")
        return

def main():
    parser = argparse.ArgumentParser(description='Monitor Stage 1 training')
    parser.add_argument('--output-dir', type=str, 
                       default='/root/autodl-tmp/text2Mol-stage1',
                       help='Training output directory')
    parser.add_argument('--interval', type=int, default=5,
                       help='Refresh interval in seconds')
    
    args = parser.parse_args()
    monitor(args.output_dir, args.interval)

if __name__ == "__main__":
    main()