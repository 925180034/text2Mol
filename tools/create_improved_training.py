#!/usr/bin/env python3
"""
创建改进的训练脚本，解决磁盘空间问题
"""

from pathlib import Path

def create_improved_script():
    """创建改进的safe_background_training.py"""
    
    improved_script = '''#!/usr/bin/env python3
"""
改进的后台训练脚本 - 增强磁盘空间管理
"""

import os
import sys
import subprocess
import time
import signal
import shutil
from pathlib import Path
import psutil
import datetime
import threading

class ImprovedSafeTrainingManager:
    def __init__(self):
        self.base_output_dir = "/root/autodl-tmp/text2Mol-outputs/safe_training"
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # 改进的配置
        self.CHECK_INTERVAL = 60  # 从5分钟改为1分钟
        self.MAX_CHECKPOINTS = 3  # 最多保留3个checkpoint
        self.CHECKPOINT_FREQUENCY = 2000  # 每2000步保存一次(原来是1000步)
        self.monitoring = True
        
    def get_disk_info(self):
        """获取磁盘使用信息"""
        disk_usage = shutil.disk_usage("/root/autodl-tmp")
        return {
            'total_gb': disk_usage.total / (1024**3),
            'used_gb': disk_usage.used / (1024**3),
            'free_gb': disk_usage.free / (1024**3),
            'used_percent': (disk_usage.used / disk_usage.total) * 100
        }
    
    def cleanup_checkpoints_aggressive(self, modality_dir):
        """更激进的checkpoint清理策略"""
        checkpoint_files = []
        
        # 查找所有checkpoint文件
        for file in Path(modality_dir).glob("*.pt"):
            if any(keyword in file.name.lower() for keyword in ['checkpoint', 'model', 'epoch']):
                stat = file.stat()
                checkpoint_files.append({
                    'path': file,
                    'size': stat.st_size,
                    'mtime': stat.st_mtime,
                    'name': file.name
                })
        
        # 按修改时间排序
        checkpoint_files.sort(key=lambda x: x['mtime'], reverse=True)
        
        # 特殊处理：保留best_model和最新的checkpoint
        essential_files = set()
        for ckpt in checkpoint_files:
            if 'best' in ckpt['name'].lower():
                essential_files.add(ckpt['path'])
                break
        
        # 保留最新的N个checkpoint
        kept_count = 0
        total_freed = 0
        
        for ckpt in checkpoint_files:
            if ckpt['path'] in essential_files:
                continue
                
            if kept_count < self.MAX_CHECKPOINTS:
                kept_count += 1
                print(f"  保留: {ckpt['name']} ({ckpt['size']/(1024**3):.1f}GB)")
            else:
                # 删除多余的文件
                try:
                    size_gb = ckpt['size'] / (1024**3)
                    ckpt['path'].unlink()
                    total_freed += size_gb
                    print(f"  删除: {ckpt['name']} ({size_gb:.1f}GB)")
                except Exception as e:
                    print(f"  删除失败: {ckpt['name']} - {e}")
        
        return total_freed
    
    def create_training_config(self, modality):
        """创建优化的训练配置"""
        config_path = Path(f"configs/safe_{modality}_config.yaml")
        config_path.parent.mkdir(exist_ok=True)
        
        # 基础配置
        base_config = {
            'smiles': {
                'batch_size': 16,
                'learning_rate': 2e-5,
                'save_frequency': 2000,  # 增加保存间隔
                'max_checkpoints': 3
            },
            'graph': {
                'batch_size': 8,
                'learning_rate': 1e-5,
                'save_frequency': 2000,
                'max_checkpoints': 3
            },
            'image': {
                'batch_size': 6,  # 减小batch size以减少内存使用
                'learning_rate': 1e-5,
                'save_frequency': 2000,
                'max_checkpoints': 3
            }
        }
        
        config_content = f"""# 安全训练配置 - {modality}模态
# 生成时间: {datetime.datetime.now()}

# 数据配置
data:
  train_file: 'Datasets/train.csv'
  val_file: 'Datasets/validation.csv'
  test_file: 'Datasets/test.csv'
  modality: '{modality}'
  batch_size: {base_config[modality]['batch_size']}
  num_workers: 4
  
# 模型配置
model:
  hidden_size: 768
  num_layers: 6
  num_heads: 12
  dropout: 0.1
  fusion_type: 'both'
  use_cached_molt5: true

# 训练配置
training:
  num_epochs: 5
  learning_rate: {base_config[modality]['learning_rate']}
  warmup_steps: 500
  gradient_accumulation_steps: 2
  max_grad_norm: 1.0
  
  # 改进的checkpoint配置
  save_frequency: {base_config[modality]['save_frequency']}
  save_total_limit: {base_config[modality]['max_checkpoints']}
  save_best_only: false  # 同时保存best和checkpoint
  
  # 磁盘空间保护
  disk_space_threshold: 5.0  # GB，低于此值停止训练
  auto_cleanup: true
  cleanup_keep_count: 3

# 日志配置
logging:
  log_every_n_steps: 100
  tensorboard: true
  wandb: false

# 性能优化
optimization:
  fp16: true  # 使用混合精度训练
  gradient_checkpointing: true  # 减少显存使用
  find_unused_parameters: false
"""
        
        config_path.write_text(config_content)
        return str(config_path)
    
    def monitor_disk_space_aggressive(self, training_pid):
        """更激进的磁盘监控"""
        last_cleanup_time = time.time()
        
        while self.monitoring:
            try:
                if not psutil.pid_exists(training_pid):
                    print("训练进程已结束")
                    break
                
                disk_info = self.get_disk_info()
                current_time = time.time()
                
                # 每分钟输出一次状态
                print(f"\\n[{datetime.datetime.now().strftime('%H:%M:%S')}] 磁盘监控:")
                print(f"  使用率: {disk_info['used_percent']:.1f}%")
                print(f"  可用空间: {disk_info['free_gb']:.1f}GB")
                
                # 多级响应策略
                if disk_info['used_percent'] > 95 or disk_info['free_gb'] < 5:
                    print("🚨 紧急！磁盘空间严重不足")
                    self.emergency_cleanup(training_pid)
                elif disk_info['used_percent'] > 90 or disk_info['free_gb'] < 10:
                    print("⚠️ 磁盘空间不足，立即清理")
                    self.perform_cleanup()
                elif disk_info['used_percent'] > 80 or disk_info['free_gb'] < 15:
                    # 每10分钟清理一次
                    if current_time - last_cleanup_time > 600:
                        print("🧹 定期清理")
                        self.perform_cleanup()
                        last_cleanup_time = current_time
                
                time.sleep(self.CHECK_INTERVAL)
                
            except Exception as e:
                print(f"监控错误: {e}")
                time.sleep(30)
    
    def perform_cleanup(self):
        """执行清理"""
        print("\\n🧹 开始清理checkpoint文件...")
        total_freed = 0
        
        for modality in ['smiles', 'graph', 'image']:
            modality_dir = Path(self.base_output_dir) / modality
            if modality_dir.exists():
                freed = self.cleanup_checkpoints_aggressive(modality_dir)
                total_freed += freed
        
        print(f"✅ 清理完成，释放了 {total_freed:.1f}GB")
        
        # 显示清理后状态
        disk_info = self.get_disk_info()
        print(f"清理后: 使用率 {disk_info['used_percent']:.1f}%, 可用 {disk_info['free_gb']:.1f}GB")
    
    def emergency_cleanup(self, training_pid):
        """紧急清理并可能停止训练"""
        print("\\n🚨 执行紧急清理...")
        self.perform_cleanup()
        
        # 再次检查
        disk_info = self.get_disk_info()
        if disk_info['free_gb'] < 3:
            print("❌ 空间仍然不足，停止训练！")
            try:
                os.kill(training_pid, signal.SIGTERM)
                print("✅ 已停止训练进程")
            except:
                pass
            self.monitoring = False
    
    def start_training(self, modality):
        """启动训练"""
        print(f"\\n🚀 启动改进的{modality}模态训练")
        print("=" * 60)
        
        # 检查初始磁盘空间
        disk_info = self.get_disk_info()
        print(f"初始磁盘状态: {disk_info['used_percent']:.1f}% 使用, {disk_info['free_gb']:.1f}GB 可用")
        
        if disk_info['free_gb'] < 10:
            print("⚠️ 可用空间不足10GB，先执行清理...")
            self.perform_cleanup()
        
        # 创建配置文件
        config_path = self.create_training_config(modality)
        
        # 准备输出目录
        output_dir = Path(self.base_output_dir) / modality
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 准备日志文件
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"improved_{modality}_{timestamp}.log"
        
        # 构建训练命令
        cmd = [
            sys.executable,
            "train_multimodal.py",
            "--config", config_path,
            "--output_dir", str(output_dir),
            "--modality", modality,
            "--checkpoint_frequency", str(self.CHECKPOINT_FREQUENCY),
            "--max_checkpoints", str(self.MAX_CHECKPOINTS)
        ]
        
        print(f"\\n执行命令: {' '.join(cmd)}")
        print(f"日志文件: {log_file}")
        print(f"配置文件: {config_path}")
        
        # 启动训练进程
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid
            )
        
        print(f"\\n✅ 训练已启动 (PID: {process.pid})")
        
        # 启动监控线程
        monitor_thread = threading.Thread(
            target=self.monitor_disk_space_aggressive,
            args=(process.pid,)
        )
        monitor_thread.daemon = True
        monitor_thread.start()
        
        print("✅ 磁盘监控已启动 (检查间隔: 1分钟)")
        print("\\n💡 改进内容:")
        print("  - 检查间隔: 5分钟 → 1分钟")
        print("  - 保存频率: 每1000步 → 每2000步")
        print("  - 最大文件数: 无限制 → 3个")
        print("  - 清理策略: 被动 → 主动")
        print("  - Batch size: 8 → 6 (Image模态)")
        
        return process.pid, log_file

def main():
    if len(sys.argv) < 2:
        print("使用方法: python improved_safe_training.py [smiles|graph|image]")
        sys.exit(1)
    
    modality = sys.argv[1].lower()
    if modality not in ['smiles', 'graph', 'image']:
        print(f"错误: 不支持的模态 '{modality}'")
        print("支持的模态: smiles, graph, image")
        sys.exit(1)
    
    manager = ImprovedSafeTrainingManager()
    pid, log_file = manager.start_training(modality)
    
    print(f"\\n训练进程PID: {pid}")
    print(f"查看日志: tail -f {log_file}")
    print(f"\\n停止训练: kill -TERM -{pid}")

if __name__ == "__main__":
    main()
'''
    
    # 保存改进的脚本
    script_path = Path("improved_safe_training.py")
    script_path.write_text(improved_script)
    script_path.chmod(0o755)
    
    print("✅ 创建了改进的训练脚本: improved_safe_training.py")
    print("\n改进内容:")
    print("1. 监控间隔: 5分钟 → 1分钟")
    print("2. 保存频率: 每1000步 → 每2000步")  
    print("3. 最大保留: 无限制 → 3个checkpoint")
    print("4. 清理策略: 被动响应 → 主动清理")
    print("5. 紧急响应: 可用<3GB时自动停止")
    print("6. Batch size优化: 减少内存压力")
    
    print("\n使用方法:")
    print("1. 先清理磁盘: python disk_cleanup_report.py --cleanup")
    print("2. 启动改进训练: python improved_safe_training.py image")

if __name__ == "__main__":
    create_improved_script()