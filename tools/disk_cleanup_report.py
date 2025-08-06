#!/usr/bin/env python3
"""
磁盘空间分析和清理报告
"""

import os
import datetime
from pathlib import Path
import subprocess

def get_disk_usage():
    """获取磁盘使用情况"""
    cmd = "df -h /root/autodl-tmp"
    result = subprocess.run(cmd.split(), capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')
    if len(lines) > 1:
        parts = lines[1].split()
        return {
            'total': parts[1],
            'used': parts[2],
            'available': parts[3],
            'percent': parts[4]
        }
    return None

def analyze_checkpoints():
    """分析checkpoint文件"""
    checkpoint_dir = Path("/root/autodl-tmp/text2Mol-outputs/safe_training/image")
    if not checkpoint_dir.exists():
        return []
    
    checkpoints = []
    for file in checkpoint_dir.glob("*.pt"):
        stat = file.stat()
        checkpoints.append({
            'name': file.name,
            'size_gb': stat.st_size / (1024**3),
            'mtime': datetime.datetime.fromtimestamp(stat.st_mtime)
        })
    
    return sorted(checkpoints, key=lambda x: x['mtime'], reverse=True)

def generate_report():
    """生成分析报告"""
    print("📊 训练状态和磁盘空间分析报告")
    print("=" * 60)
    print(f"报告时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 训练状态分析
    print("🔍 训练状态分析:")
    print("1. Image模态训练在Epoch 1/5的97%时停止")
    print("2. 停止原因: 磁盘空间已满(100%)")
    print("3. 错误类型: torch.save()失败 - 无法写入checkpoint文件")
    print("4. 最后的checkpoint: checkpoint_step_5000.pt (仅1.2GB,未完成)")
    print()
    
    # 磁盘使用情况
    disk_info = get_disk_usage()
    if disk_info:
        print("💾 磁盘使用情况 (/root/autodl-tmp):")
        print(f"  总容量: {disk_info['total']}")
        print(f"  已使用: {disk_info['used']} ({disk_info['percent']})")
        print(f"  可用: {disk_info['available']}")
        print()
    
    # Checkpoint文件分析
    checkpoints = analyze_checkpoints()
    if checkpoints:
        print("📁 Checkpoint文件分析:")
        total_size = sum(c['size_gb'] for c in checkpoints)
        print(f"  文件数量: {len(checkpoints)}")
        print(f"  总大小: {total_size:.1f}GB")
        print()
        print("  文件列表:")
        for ckpt in checkpoints:
            print(f"    - {ckpt['name']}: {ckpt['size_gb']:.1f}GB ({ckpt['mtime'].strftime('%H:%M:%S')})")
        print()
    
    # 自动清理机制分析
    print("🤔 自动清理机制分析:")
    print("1. 监控间隔: 每5分钟检查一次")
    print("2. 清理触发: 磁盘使用率>85%时")
    print("3. 清理策略: 保留最新的1个checkpoint")
    print()
    print("❌ 失败原因:")
    print("  - 训练过程生成checkpoint太快(每10分钟5.6GB)")
    print("  - 5分钟监控间隔来不及响应")
    print("  - 第1个epoch就生成了8个checkpoint(约40GB)")
    print()
    
    # 解决方案
    print("✅ 解决方案:")
    print("1. 立即清理: 删除多余的checkpoint文件")
    print("2. 调整策略: 减少checkpoint保存频率")
    print("3. 优化监控: 缩短监控间隔到1分钟")
    print("4. 限制数量: 最多保留2-3个checkpoint")
    print()
    
    print("📝 回答您的问题:")
    print("1. 训练未完成,在第1个epoch的97%时因磁盘满而停止")
    print("2. 自动清理机制存在但未及时触发")
    print("3. 需要手动清理后重启训练")

def cleanup_checkpoints(keep_count=2):
    """清理多余的checkpoint文件"""
    print("\n🧹 开始清理checkpoint文件...")
    
    checkpoint_dir = Path("/root/autodl-tmp/text2Mol-outputs/safe_training/image")
    checkpoints = analyze_checkpoints()
    
    if len(checkpoints) <= keep_count:
        print(f"✅ 只有{len(checkpoints)}个文件,无需清理")
        return 0
    
    total_freed = 0
    files_to_delete = checkpoints[keep_count:]
    
    for ckpt in files_to_delete:
        file_path = checkpoint_dir / ckpt['name']
        try:
            file_path.unlink()
            total_freed += ckpt['size_gb']
            print(f"  ✅ 删除: {ckpt['name']} ({ckpt['size_gb']:.1f}GB)")
        except Exception as e:
            print(f"  ❌ 删除失败: {ckpt['name']} - {e}")
    
    print(f"\n✅ 清理完成! 释放了 {total_freed:.1f}GB 空间")
    
    # 显示清理后的磁盘状态
    disk_info = get_disk_usage()
    if disk_info:
        print(f"\n清理后磁盘使用: {disk_info['percent']} (可用: {disk_info['available']})")
    
    return total_freed

if __name__ == "__main__":
    generate_report()
    
    # 询问是否清理
    print("\n" + "=" * 60)
    print("💡 建议立即清理以释放空间")
    print("运行: python disk_cleanup_report.py --cleanup")
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--cleanup":
        cleanup_checkpoints(keep_count=2)
        print("\n🎉 清理完成! 现在可以重启训练了")
        print("运行: python safe_background_training.py image")