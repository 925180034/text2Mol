#!/usr/bin/env python3
"""
Real-time training monitor for Stage 2 nine-modality training
Shows progress for all 9 input-output combinations
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
            ['du', '-sh', '/root/autodl-tmp/text2Mol-stage2'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            size = result.stdout.split()[0]
            
            # Also check total autodl-tmp usage
            result2 = subprocess.run(
                ['du', '-sh', '/root/autodl-tmp'],
                capture_output=True, text=True
            )
            if result2.returncode == 0:
                total_size = result2.stdout.split()[0]
                return f"{size} / {total_size} total"
    except:
        pass
    return "N/A"

def parse_modality_stats(log_file):
    """Parse modality-specific performance from log"""
    modality_stats = {}
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines[-100:]:  # Check last 100 lines
            if "Modality Performance:" in line:
                # Parse the following lines for modality stats
                idx = lines.index(line)
                for i in range(idx+1, min(idx+10, len(lines))):
                    if ":" in lines[i] and "_" in lines[i]:
                        parts = lines[i].strip().split(":")
                        if len(parts) == 2:
                            mod_name = parts[0].strip()
                            mod_loss = parts[1].strip()
                            modality_stats[mod_name] = mod_loss
    except:
        pass
    return modality_stats

def tail_log(log_file, n=10):
    """Get last n lines of log file"""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            return lines[-n:]
    except:
        return []

def format_modality_grid(stats):
    """Format 9-modality performance as a 3x3 grid"""
    modalities = ['smiles', 'graph', 'image']
    grid = []
    
    grid.append("    Input↓ / Output→ | SMILES | GRAPH  | IMAGE")
    grid.append("    ------------------|--------|--------|-------")
    
    for input_mod in modalities:
        row = f"    {input_mod:17} |"
        for output_mod in modalities:
            key = f"{input_mod}_{output_mod}"
            if key in stats:
                value = stats[key][:6] if len(stats[key]) > 6 else stats[key].ljust(6)
                row += f" {value} |"
            else:
                row += "   -    |"
        grid.append(row)
    
    return "\n".join(grid)

def monitor(output_dir='/root/autodl-tmp/text2Mol-stage2', refresh_interval=5):
    """Monitor Stage 2 training progress"""
    output_dir = Path(output_dir)
    
    print("=" * 70)
    print("📊 STAGE 2 NINE-MODALITY TRAINING MONITOR")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Refresh interval: {refresh_interval}s")
    print("Press Ctrl+C to exit\n")
    
    try:
        while True:
            # Clear screen
            os.system('clear')
            
            # Header
            print("=" * 70)
            print(f"📊 STAGE 2 MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 70)
            
            # GPU Status
            gpu_info = get_gpu_info()
            if gpu_info:
                print("\n🖥️ GPU Status:")
                print(f"  Utilization: {gpu_info['gpu_util']}")
                print(f"  Memory: {gpu_info['memory_used']} / {gpu_info['memory_total']}")
                print(f"  Temperature: {gpu_info['temperature']}")
            
            # Disk Usage
            print(f"\n💾 Model Storage:")
            print(f"  Stage 2: {get_disk_usage()}")
            
            # Find latest log file
            log_files = list(output_dir.glob('training_*.log'))
            if log_files:
                latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
                
                print(f"\n📝 Latest Log: {latest_log.name}")
                print("-" * 70)
                
                # Show last few lines of log
                last_lines = tail_log(latest_log, n=10)
                for line in last_lines:
                    print(line.rstrip())
                
                # Parse and display modality statistics
                modality_stats = parse_modality_stats(latest_log)
                if modality_stats:
                    print("\n🎯 Nine-Modality Performance Grid:")
                    print(format_modality_grid(modality_stats))
            else:
                print("\n⏳ Waiting for training to start...")
            
            # Check for checkpoints
            checkpoints = list(output_dir.glob('*.pt'))
            if checkpoints:
                print(f"\n📦 Saved Models ({len(checkpoints)} total):")
                
                # Sort by modification time
                checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                
                # Show latest 3 models
                for ckpt in checkpoints[:3]:
                    size_gb = ckpt.stat().st_size / (1024**3)
                    mtime = datetime.fromtimestamp(ckpt.stat().st_mtime)
                    print(f"  {ckpt.name}: {size_gb:.1f}GB (saved {mtime.strftime('%H:%M:%S')})")
                
                # Calculate total storage
                total_size_gb = sum(c.stat().st_size for c in checkpoints) / (1024**3)
                print(f"\n  Total storage: {total_size_gb:.1f}GB / 50GB limit")
                
                if total_size_gb > 45:
                    print("  ⚠️ WARNING: Approaching storage limit!")
            
            # Check if training is still running
            try:
                result = subprocess.run(['pgrep', '-f', 'train_stage2_nine_modality.py'], 
                                      capture_output=True)
                if result.returncode == 0:
                    pid = result.stdout.decode().strip().split('\n')[0]
                    print(f"\n✅ Training is running (PID: {pid})")
                    
                    # Estimate progress
                    if log_files:
                        # Try to parse epoch info
                        for line in last_lines:
                            if "Epoch" in line and "/" in line:
                                print(f"📈 Progress: {line.strip()}")
                                break
                else:
                    print("\n⚠️ Training process not found")
            except:
                pass
            
            print("\n" + "=" * 70)
            print("Press Ctrl+C to exit monitor")
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitor stopped")
        return

def main():
    parser = argparse.ArgumentParser(description='Monitor Stage 2 nine-modality training')
    parser.add_argument('--output-dir', type=str, 
                       default='/root/autodl-tmp/text2Mol-stage2',
                       help='Training output directory')
    parser.add_argument('--interval', type=int, default=5,
                       help='Refresh interval in seconds')
    
    args = parser.parse_args()
    monitor(args.output_dir, args.interval)

if __name__ == "__main__":
    main()