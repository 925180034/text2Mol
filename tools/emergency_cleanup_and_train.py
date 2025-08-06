#!/usr/bin/env python3
"""
紧急清理和多GPU并行训练系统
"""

import os
import sys
import shutil
import subprocess
import time
import threading
import signal
from pathlib import Path
import datetime
import psutil
import torch

class MultiModalTrainingManager:
    def __init__(self):
        self.base_output_dir = "/root/autodl-tmp/text2Mol-outputs/safe_training"
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # 训练配置
        self.modalities = ['smiles', 'graph', 'image']
        self.gpu_assignments = {
            'smiles': 0,
            'graph': 1, 
            'image': 0  # SMILES训练完成后使用GPU 0
        }
        
        # 激进的清理配置
        self.MAX_CHECKPOINTS_PER_MODALITY = 2  # 每个模态最多保留2个checkpoint
        self.CHECK_INTERVAL = 30  # 每30秒检查一次
        self.DISK_THRESHOLD = 85  # 磁盘使用超过85%就清理
        self.MIN_FREE_GB = 10  # 保持至少10GB可用空间
        
        self.monitoring = True
        self.training_processes = {}
        
    def get_disk_info(self):
        """获取磁盘使用信息"""
        disk_usage = shutil.disk_usage("/root/autodl-tmp")
        return {
            'total_gb': disk_usage.total / (1024**3),
            'used_gb': disk_usage.used / (1024**3),
            'free_gb': disk_usage.free / (1024**3),
            'used_percent': (disk_usage.used / disk_usage.total) * 100
        }
    
    def emergency_cleanup(self):
        """紧急清理所有多余的checkpoint"""
        print("\n🚨 执行紧急磁盘清理...")
        total_freed = 0
        
        for modality in self.modalities:
            modality_dir = Path(self.base_output_dir) / modality
            if not modality_dir.exists():
                continue
                
            print(f"\n清理 {modality} 模态:")
            
            # 获取所有checkpoint文件
            checkpoints = []
            for file in modality_dir.glob("*.pt"):
                stat = file.stat()
                checkpoints.append({
                    'path': file,
                    'name': file.name,
                    'size': stat.st_size,
                    'mtime': stat.st_mtime
                })
            
            # 按修改时间排序
            checkpoints.sort(key=lambda x: x['mtime'], reverse=True)
            
            # 保留策略
            keep_files = set()
            
            # 1. 保留最新的checkpoint
            if checkpoints:
                keep_files.add(checkpoints[0]['path'])
            
            # 2. 保留best_model
            for ckpt in checkpoints:
                if 'best' in ckpt['name'].lower():
                    keep_files.add(ckpt['path'])
                    break
            
            # 3. 如果空间允许，再保留一个
            if len(checkpoints) > 2 and len(keep_files) < self.MAX_CHECKPOINTS_PER_MODALITY:
                for ckpt in checkpoints[1:]:
                    if ckpt['path'] not in keep_files:
                        keep_files.add(ckpt['path'])
                        break
            
            # 删除其他文件
            for ckpt in checkpoints:
                if ckpt['path'] not in keep_files:
                    try:
                        size_gb = ckpt['size'] / (1024**3)
                        ckpt['path'].unlink()
                        total_freed += size_gb
                        print(f"  ❌ 删除: {ckpt['name']} ({size_gb:.1f}GB)")
                    except Exception as e:
                        print(f"  ⚠️ 删除失败: {ckpt['name']} - {e}")
                else:
                    size_gb = ckpt['size'] / (1024**3)
                    print(f"  ✅ 保留: {ckpt['name']} ({size_gb:.1f}GB)")
        
        print(f"\n✅ 清理完成，释放了 {total_freed:.1f}GB")
        
        # 显示清理后状态
        disk_info = self.get_disk_info()
        print(f"清理后: {disk_info['used_percent']:.1f}% 使用, {disk_info['free_gb']:.1f}GB 可用")
        
        return total_freed
    
    def continuous_monitoring(self):
        """持续监控磁盘并自动清理"""
        print("\n🔍 启动持续磁盘监控...")
        
        while self.monitoring:
            try:
                disk_info = self.get_disk_info()
                
                # 每30秒显示状态
                print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] 磁盘监控:")
                print(f"  使用: {disk_info['used_percent']:.1f}% | 可用: {disk_info['free_gb']:.1f}GB")
                
                # 检查活跃的训练进程
                active_count = sum(1 for pid in self.training_processes.values() 
                                 if pid and psutil.pid_exists(pid))
                print(f"  活跃训练: {active_count}/{len(self.training_processes)}")
                
                # 触发清理的条件
                if (disk_info['used_percent'] > self.DISK_THRESHOLD or 
                    disk_info['free_gb'] < self.MIN_FREE_GB):
                    print(f"  ⚠️ 触发自动清理 (使用率>{self.DISK_THRESHOLD}% 或 可用<{self.MIN_FREE_GB}GB)")
                    self.emergency_cleanup()
                
                time.sleep(self.CHECK_INTERVAL)
                
            except Exception as e:
                print(f"监控错误: {e}")
                time.sleep(10)
    
    def create_optimized_config(self, modality, gpu_id):
        """创建优化的训练配置"""
        config_path = Path(f"configs/multi_gpu_{modality}_config.yaml")
        config_path.parent.mkdir(exist_ok=True)
        
        # 针对双GPU优化的配置
        configs = {
            'smiles': {
                'batch_size': 24,  # 增大batch size
                'gradient_accumulation': 1,
                'learning_rate': 3e-5,
                'save_frequency': 3000,  # 减少保存频率
            },
            'graph': {
                'batch_size': 16,
                'gradient_accumulation': 2,
                'learning_rate': 2e-5,
                'save_frequency': 2500,
            },
            'image': {
                'batch_size': 8,
                'gradient_accumulation': 4,
                'learning_rate': 1e-5,
                'save_frequency': 2000,
            }
        }
        
        config = configs[modality]
        
        config_content = f"""# 多GPU训练配置 - {modality}模态 (GPU {gpu_id})
# 生成时间: {datetime.datetime.now()}

data:
  train_file: 'Datasets/train.csv'
  val_file: 'Datasets/validation.csv'
  test_file: 'Datasets/test.csv'
  modality: '{modality}'
  batch_size: {config['batch_size']}
  num_workers: 4
  pin_memory: true
  prefetch_factor: 2

model:
  hidden_size: 768
  num_layers: 6
  num_heads: 12
  dropout: 0.1
  fusion_type: 'both'
  use_cached_molt5: true

training:
  num_epochs: 5
  learning_rate: {config['learning_rate']}
  warmup_steps: 500
  gradient_accumulation_steps: {config['gradient_accumulation']}
  max_grad_norm: 1.0
  
  # 优化的checkpoint策略
  save_frequency: {config['save_frequency']}
  save_total_limit: 2  # 只保留2个checkpoint
  save_best_only: false
  save_on_epoch_end: true
  
  # 磁盘保护
  disk_space_threshold: 5.0
  auto_cleanup: true
  cleanup_keep_count: 2

optimization:
  fp16: true  # 混合精度训练
  gradient_checkpointing: true
  find_unused_parameters: false
  dataloader_drop_last: true
  
device:
  gpu_id: {gpu_id}
  
logging:
  log_every_n_steps: 100
  tensorboard: true
  wandb: false
"""
        
        config_path.write_text(config_content)
        return str(config_path)
    
    def start_modality_training(self, modality, gpu_id):
        """启动单个模态的训练"""
        print(f"\n🚀 在GPU {gpu_id} 上启动 {modality} 模态训练")
        
        # 创建配置
        config_path = self.create_optimized_config(modality, gpu_id)
        
        # 准备输出目录
        output_dir = Path(self.base_output_dir) / modality
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 日志文件
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"multi_gpu_{modality}_{timestamp}.log"
        
        # 构建命令
        cmd = [
            sys.executable,
            "train_multimodal.py",
            "--config", config_path,
            "--output_dir", str(output_dir),
            "--modality", modality,
        ]
        
        # 设置环境变量指定GPU
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        print(f"执行命令: {' '.join(cmd)}")
        print(f"使用GPU: {gpu_id}")
        print(f"日志文件: {log_file}")
        
        # 启动进程
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setsid
            )
        
        self.training_processes[modality] = process.pid
        print(f"✅ {modality} 训练已启动 (PID: {process.pid})")
        
        return process.pid
    
    def start_all_training(self):
        """启动所有模态的训练"""
        print("\n🎯 启动多模态并行训练系统")
        print("=" * 60)
        
        # 首先清理磁盘
        disk_info = self.get_disk_info()
        print(f"初始磁盘状态: {disk_info['used_percent']:.1f}% 使用, {disk_info['free_gb']:.1f}GB 可用")
        
        if disk_info['free_gb'] < 15:
            print("⚠️ 磁盘空间不足，执行清理...")
            self.emergency_cleanup()
        
        # 启动监控线程
        monitor_thread = threading.Thread(target=self.continuous_monitoring)
        monitor_thread.daemon = True
        monitor_thread.start()
        print("✅ 磁盘监控已启动")
        
        # 检查GPU
        gpu_count = torch.cuda.device_count()
        print(f"\n检测到 {gpu_count} 个GPU")
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # 启动训练
        print("\n开始启动训练任务:")
        
        # 1. 在GPU 0上启动SMILES
        self.start_modality_training('smiles', 0)
        time.sleep(5)  # 等待进程启动
        
        # 2. 在GPU 1上启动Graph  
        self.start_modality_training('graph', 1)
        time.sleep(5)
        
        # 3. 在GPU 0上启动Image（与SMILES共享）
        self.start_modality_training('image', 0)
        
        print("\n✅ 所有训练任务已启动!")
        print("\n监控命令:")
        print("  查看GPU使用: nvidia-smi -l 1")
        print("  查看日志: tail -f logs/multi_gpu_*.log")
        print("  停止所有训练: python emergency_cleanup_and_train.py --stop")
        
    def stop_all_training(self):
        """停止所有训练"""
        print("\n🛑 停止所有训练进程...")
        
        for modality, pid in self.training_processes.items():
            if pid and psutil.pid_exists(pid):
                try:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                    print(f"  ✅ 停止 {modality} (PID: {pid})")
                except:
                    print(f"  ⚠️ 无法停止 {modality} (PID: {pid})")
        
        self.monitoring = False
        print("✅ 所有训练已停止")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--stop':
        # 停止模式
        manager = MultiModalTrainingManager()
        manager.stop_all_training()
    else:
        # 启动模式
        manager = MultiModalTrainingManager()
        
        # 首先执行紧急清理
        print("🧹 首先执行紧急清理...")
        manager.emergency_cleanup()
        
        # 然后启动训练
        manager.start_all_training()
        
        # 保持主进程运行
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("\n⚠️ 收到中断信号")
            manager.stop_all_training()

if __name__ == "__main__":
    main()