#!/usr/bin/env python3
"""
快速清理脚本
"""

import os
from pathlib import Path

def quick_cleanup():
    base_dir = Path("/root/autodl-tmp/text2Mol-outputs/safe_training")
    total_freed = 0
    
    print("🧹 执行快速清理...")
    
    # 清理每个模态
    for modality in ['smiles', 'graph', 'image']:
        modality_dir = base_dir / modality
        if not modality_dir.exists():
            continue
            
        print(f"\n清理 {modality}:")
        
        # 获取所有pt文件
        pt_files = list(modality_dir.glob("*.pt"))
        pt_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # 保留最新的2个文件
        keep_count = 2
        for i, file in enumerate(pt_files):
            size_gb = file.stat().st_size / (1024**3)
            if i < keep_count:
                print(f"  ✅ 保留: {file.name} ({size_gb:.1f}GB)")
            else:
                try:
                    file.unlink()
                    total_freed += size_gb
                    print(f"  ❌ 删除: {file.name} ({size_gb:.1f}GB)")
                except Exception as e:
                    print(f"  ⚠️ 删除失败: {file.name} - {e}")
    
    print(f"\n✅ 共释放 {total_freed:.1f}GB")
    
    # 显示磁盘状态
    import shutil
    disk = shutil.disk_usage("/root/autodl-tmp")
    print(f"\n磁盘状态: {disk.used/(1024**3):.1f}GB/{disk.total/(1024**3):.1f}GB ({disk.used/disk.total*100:.1f}%)")
    print(f"可用空间: {disk.free/(1024**3):.1f}GB")

if __name__ == "__main__":
    quick_cleanup()