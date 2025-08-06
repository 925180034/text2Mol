#!/usr/bin/env python3
"""
磁盘守护进程 - 确保训练不会因为磁盘满而失败
"""

import os
import time
import shutil
from pathlib import Path
import datetime
import threading
import signal
import sys

class DiskGuardian:
    def __init__(self):
        self.running = True
        self.base_dir = Path("/root/autodl-tmp/text2Mol-outputs/safe_training")
        
        # 激进的清理策略
        self.CRITICAL_THRESHOLD = 90  # 90%时紧急清理
        self.WARNING_THRESHOLD = 80   # 80%时常规清理
        self.TARGET_FREE_GB = 15      # 目标保持15GB可用
        self.CHECK_INTERVAL = 30      # 每30秒检查一次
        self.MAX_CHECKPOINTS = 2      # 每个模态最多2个checkpoint
        
    def signal_handler(self, sig, frame):
        """处理中断信号"""
        print("\n🛑 停止磁盘守护进程...")
        self.running = False
        sys.exit(0)
    
    def get_disk_info(self):
        """获取磁盘信息"""
        disk = shutil.disk_usage("/root/autodl-tmp")
        return {
            'used_gb': disk.used / (1024**3),
            'free_gb': disk.free / (1024**3),
            'total_gb': disk.total / (1024**3),
            'used_percent': (disk.used / disk.total) * 100
        }
    
    def cleanup_modality(self, modality_dir, keep_count=2):
        """清理单个模态的checkpoint"""
        if not modality_dir.exists():
            return 0
        
        freed = 0
        pt_files = list(modality_dir.glob("*.pt"))
        
        if len(pt_files) <= keep_count:
            return 0
        
        # 按时间排序
        pt_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # 特殊文件保护
        protected = set()
        for f in pt_files:
            if 'best' in f.name.lower():
                protected.add(f)
        
        # 删除多余文件
        kept = 0
        for f in pt_files:
            if f in protected:
                continue
                
            if kept < keep_count:
                kept += 1
            else:
                try:
                    size = f.stat().st_size / (1024**3)
                    f.unlink()
                    freed += size
                    print(f"    删除: {f.name} ({size:.1f}GB)")
                except Exception as e:
                    print(f"    删除失败: {f.name} - {e}")
        
        return freed
    
    def emergency_cleanup(self):
        """紧急清理 - 更激进"""
        print(f"\n🚨 [{datetime.datetime.now().strftime('%H:%M:%S')}] 执行紧急清理")
        
        total_freed = 0
        
        # 清理所有模态，只保留1个checkpoint
        for modality in ['smiles', 'graph', 'image']:
            modality_dir = self.base_dir / modality
            print(f"  清理 {modality}:")
            freed = self.cleanup_modality(modality_dir, keep_count=1)
            total_freed += freed
        
        # 清理tensorboard日志
        for modality in ['smiles', 'graph', 'image']:
            tb_dir = self.base_dir / modality / 'tensorboard'
            if tb_dir.exists():
                try:
                    shutil.rmtree(tb_dir)
                    print(f"  清理tensorboard日志: {modality}")
                except:
                    pass
        
        print(f"  ✅ 共释放 {total_freed:.1f}GB")
        return total_freed
    
    def regular_cleanup(self):
        """常规清理"""
        print(f"\n🧹 [{datetime.datetime.now().strftime('%H:%M:%S')}] 执行常规清理")
        
        total_freed = 0
        
        for modality in ['smiles', 'graph', 'image']:
            modality_dir = self.base_dir / modality
            freed = self.cleanup_modality(modality_dir, keep_count=self.MAX_CHECKPOINTS)
            total_freed += freed
        
        if total_freed > 0:
            print(f"  ✅ 共释放 {total_freed:.1f}GB")
        
        return total_freed
    
    def monitor_loop(self):
        """主监控循环"""
        print("🛡️ 磁盘守护进程已启动")
        print(f"配置: 检查间隔={self.CHECK_INTERVAL}秒, 警告={self.WARNING_THRESHOLD}%, 紧急={self.CRITICAL_THRESHOLD}%")
        print("按 Ctrl+C 停止\n")
        
        last_cleanup = 0
        
        while self.running:
            try:
                disk = self.get_disk_info()
                
                # 显示状态
                status = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
                status += f"磁盘: {disk['used_percent']:.1f}% | "
                status += f"可用: {disk['free_gb']:.1f}GB"
                
                # 检查是否需要清理
                current_time = time.time()
                
                if disk['used_percent'] > self.CRITICAL_THRESHOLD or disk['free_gb'] < 5:
                    # 紧急情况
                    print(f"\n{status} 🚨 紧急!")
                    self.emergency_cleanup()
                    last_cleanup = current_time
                    
                elif disk['used_percent'] > self.WARNING_THRESHOLD or disk['free_gb'] < self.TARGET_FREE_GB:
                    # 常规清理 (最多每5分钟一次)
                    if current_time - last_cleanup > 300:
                        print(f"\n{status} ⚠️ 需要清理")
                        self.regular_cleanup()
                        last_cleanup = current_time
                    else:
                        print(f"\r{status}", end='', flush=True)
                else:
                    # 正常状态
                    print(f"\r{status} ✅", end='', flush=True)
                
                time.sleep(self.CHECK_INTERVAL)
                
            except Exception as e:
                print(f"\n❌ 监控错误: {e}")
                time.sleep(10)
    
    def start(self):
        """启动守护进程"""
        # 注册信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # 启动前先做一次清理
        disk = self.get_disk_info()
        print(f"初始状态: {disk['used_percent']:.1f}% 使用, {disk['free_gb']:.1f}GB 可用")
        
        if disk['used_percent'] > self.WARNING_THRESHOLD:
            self.regular_cleanup()
        
        # 开始监控
        self.monitor_loop()

def main():
    guardian = DiskGuardian()
    guardian.start()

if __name__ == "__main__":
    main()