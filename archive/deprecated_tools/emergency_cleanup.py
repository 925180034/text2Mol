#!/usr/bin/env python3
"""
紧急磁盘清理脚本
清理重复的checkpoint和不必要的文件
"""

import os
import shutil
from pathlib import Path

def emergency_cleanup():
    print("🚨 紧急磁盘清理...")
    
    # 1. 清理重复的checkpoint（保留最新的）
    print("\n1. 清理重复checkpoints...")
    
    # 删除旧的训练checkpoint目录
    old_dirs = [
        "/root/autodl-tmp/safe_fast_checkpoints",
        "/root/autodl-tmp/continued_checkpoints", 
        "/root/autodl-tmp/text2Mol-outputs/fast_smiles"
    ]
    
    total_freed = 0
    for dir_path in old_dirs:
        if os.path.exists(dir_path):
            # 计算大小
            size = sum(f.stat().st_size for f in Path(dir_path).rglob('*') if f.is_file())
            size_gb = size / (1024**3)
            
            print(f"  删除 {dir_path} ({size_gb:.1f}GB)")
            shutil.rmtree(dir_path)
            total_freed += size_gb
    
    # 2. 清理bg_smiles目录中的重复文件
    print("\n2. 清理bg_smiles重复文件...")
    bg_smiles_dir = "/root/autodl-tmp/text2Mol-outputs/bg_smiles"
    
    if os.path.exists(bg_smiles_dir):
        # 只保留best_model.pt，删除其他重复文件
        files_to_remove = [
            "checkpoint_step_1000.pt",
            "model_best.pt",  # 与best_model.pt重复
            "epoch_1.pt"      # 中间文件
        ]
        
        for filename in files_to_remove:
            filepath = os.path.join(bg_smiles_dir, filename)
            if os.path.exists(filepath):
                size = os.path.getsize(filepath) / (1024**3)
                print(f"  删除 {filename} ({size:.1f}GB)")
                os.remove(filepath)
                total_freed += size
    
    # 3. 删除备份文件
    print("\n3. 清理备份文件...")
    backup_files = [
        "/root/autodl-tmp/best_model_backup.pt"
    ]
    
    for file_path in backup_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024**3)
            print(f"  删除 {os.path.basename(file_path)} ({size:.1f}GB)")
            os.remove(file_path)
            total_freed += size
    
    print(f"\n✅ 清理完成！释放了 {total_freed:.1f}GB 空间")
    
    # 检查清理后的空间
    result = os.statvfs("/root/autodl-tmp")
    available_gb = (result.f_bavail * result.f_frsize) / (1024**3)
    total_gb = (result.f_blocks * result.f_frsize) / (1024**3)
    used_gb = total_gb - available_gb
    
    print(f"\n📊 清理后磁盘状态:")
    print(f"  总容量: {total_gb:.1f}GB")
    print(f"  已使用: {used_gb:.1f}GB")
    print(f"  可用空间: {available_gb:.1f}GB ({available_gb/total_gb*100:.1f}%)")

if __name__ == "__main__":
    emergency_cleanup()