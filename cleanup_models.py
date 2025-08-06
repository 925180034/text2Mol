#!/usr/bin/env python3
"""
清理模型文件，只保留最终的好模型
"""
import os
import shutil
from pathlib import Path

def cleanup_model_checkpoints():
    """清理中间checkpoint，只保留最终模型"""
    
    model_dir = Path('/root/autodl-tmp/text2Mol-outputs/fast_training')
    
    # 定义要保留的文件
    files_to_keep = {
        'smiles/final_model.pt',
        'graph/checkpoint_step_5000.pt',  # 最新的graph checkpoint
        'image/best_model.pt',  # 最好的image模型
    }
    
    print("="*70)
    print("🧹 清理模型checkpoint文件")
    print("="*70)
    
    total_freed = 0
    
    # 清理每个模态目录
    for modality in ['smiles', 'graph', 'image']:
        modality_dir = model_dir / modality
        if not modality_dir.exists():
            continue
            
        print(f"\n📂 处理 {modality} 目录:")
        
        # 列出所有.pt和.pth文件
        checkpoint_files = list(modality_dir.glob('*.pt')) + list(modality_dir.glob('*.pth'))
        
        for file_path in checkpoint_files:
            relative_path = f"{modality}/{file_path.name}"
            
            if relative_path in files_to_keep:
                print(f"  ✅ 保留: {file_path.name} ({file_path.stat().st_size / 1e9:.2f}GB)")
            else:
                file_size = file_path.stat().st_size
                print(f"  🗑️ 删除: {file_path.name} ({file_size / 1e9:.2f}GB)")
                file_path.unlink()
                total_freed += file_size
    
    # 清理其他冗余文件
    print("\n📂 清理其他冗余文件:")
    
    # 清理logs目录中的旧日志
    logs_dir = Path('/root/text2Mol/scaffold-mol-generation/logs')
    if logs_dir.exists():
        log_files = list(logs_dir.glob('*.log'))
        for log_file in log_files:
            # 保留最近的几个日志
            if 'final' not in log_file.name and 'resume' not in log_file.name:
                print(f"  🗑️ 删除日志: {log_file.name}")
                log_file.unlink()
    
    print("\n" + "="*70)
    print(f"✅ 清理完成！")
    print(f"💾 释放空间: {total_freed / 1e9:.2f}GB")
    print("="*70)
    
    return total_freed

if __name__ == "__main__":
    freed_space = cleanup_model_checkpoints()
    
    # 显示磁盘使用情况
    print("\n📊 当前磁盘使用情况:")
    os.system("df -h /root/autodl-tmp")