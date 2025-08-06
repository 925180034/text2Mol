#!/usr/bin/env python3
"""
后台训练启动器
支持多模态训练的后台执行和实时监控
"""

import subprocess
import time
import os
import json
import signal
import sys
from datetime import datetime
from pathlib import Path

class BackgroundTrainer:
    def __init__(self):
        self.pids_file = "logs/training_pids.json"
        self.status_file = "logs/training_status.json"
        self.ensure_dirs()
    
    def ensure_dirs(self):
        """确保必要目录存在"""
        Path("logs").mkdir(exist_ok=True)
        Path("/root/autodl-tmp/text2Mol-outputs").mkdir(parents=True, exist_ok=True)
    
    def get_optimal_batch_size(self, modality):
        """获取32GB显卡的最优batch size"""
        sizes = {
            'smiles': 20,
            'graph': 12, 
            'image': 8
        }
        return sizes.get(modality, 8)
    
    def start_training(self, modality, epochs=5, background=True):
        """启动训练任务"""
        batch_size = self.get_optimal_batch_size(modality)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 训练命令
        cmd = [
            'python', 'train_multimodal.py',
            '--train-data', 'Datasets/train.csv',
            '--val-data', 'Datasets/validation.csv',
            '--test-data', 'Datasets/test.csv',
            '--output-dir', f'/root/autodl-tmp/text2Mol-outputs/bg_{modality}',
            '--batch-size', str(batch_size),
            '--epochs', str(epochs),
            '--lr', '1e-4',
            '--scaffold-modality', modality,
            '--device', 'cuda'
        ]
        
        # 环境变量优化
        env = os.environ.copy()
        env.update({
            'CUDA_VISIBLE_DEVICES': '0',
            'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:2048',
            'TORCH_CUDNN_V8_API_ENABLED': '1',
        })
        
        # 日志文件
        log_file = f"logs/bg_{modality}_{timestamp}.log"
        
        print(f"🚀 启动{modality.upper()}模态后台训练...")
        print(f"   批次大小: {batch_size}")
        print(f"   训练轮数: {epochs}")
        print(f"   日志文件: {log_file}")
        
        if background:
            # 后台启动
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env,
                    preexec_fn=os.setsid  # 创建新进程组
                )
            
            pid = process.pid
            print(f"   后台PID: {pid}")
            
            # 保存PID信息
            self.save_pid_info(modality, pid, log_file, timestamp)
            
            return pid, log_file
            
        else:
            # 前台运行
            process = subprocess.Popen(cmd, env=env)
            return process.pid, log_file
    
    def save_pid_info(self, modality, pid, log_file, timestamp):
        """保存PID信息到文件"""
        pids = self.load_pids()
        pids[modality] = {
            'pid': pid,
            'log_file': log_file,
            'start_time': timestamp,
            'status': 'running'
        }
        
        with open(self.pids_file, 'w') as f:
            json.dump(pids, f, indent=2)
    
    def load_pids(self):
        """加载PID信息"""
        if os.path.exists(self.pids_file):
            try:
                with open(self.pids_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def check_process_status(self, pid):
        """检查进程状态"""
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    
    def get_training_status(self):
        """获取所有训练任务状态"""
        pids = self.load_pids()
        status = {}
        
        for modality, info in pids.items():
            pid = info['pid']
            is_running = self.check_process_status(pid)
            
            status[modality] = {
                'pid': pid,
                'running': is_running,
                'log_file': info['log_file'],
                'start_time': info['start_time'],
                'status': 'running' if is_running else 'completed'
            }
        
        return status
    
    def stop_training(self, modality=None):
        """停止训练任务"""
        pids = self.load_pids()
        
        if modality:
            # 停止特定模态
            if modality in pids:
                pid = pids[modality]['pid']
                try:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                    print(f"✅ 已停止{modality}训练 (PID: {pid})")
                except:
                    print(f"❌ 无法停止{modality}训练")
            else:
                print(f"❌ 未找到{modality}训练任务")
        else:
            # 停止所有训练
            for mod, info in pids.items():
                pid = info['pid']
                try:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                    print(f"✅ 已停止{mod}训练 (PID: {pid})")
                except:
                    print(f"❌ 无法停止{mod}训练")
    
    def start_sequential_training(self):
        """启动顺序后台训练"""
        print("🔥 启动顺序后台训练")
        print("三个模态将依次在后台训练")
        
        modalities = ['smiles', 'graph', 'image']
        
        for modality in modalities:
            pid, log_file = self.start_training(modality, epochs=5, background=True)
            time.sleep(2)  # 短暂延迟避免冲突
        
        print(f"\n✅ 所有训练任务已启动")
        print("使用以下命令监控:")
        print("python background_training.py --monitor")
    
    def start_single_training(self, modality):
        """启动单个模态后台训练"""
        pid, log_file = self.start_training(modality, epochs=5, background=True)
        print(f"\n✅ {modality}训练已在后台启动")
        print("使用以下命令监控:")
        print("python background_training.py --monitor")

def main():
    trainer = BackgroundTrainer()
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == '--monitor':
            # 启动监控
            from training_monitor import TrainingMonitor
            monitor = TrainingMonitor()
            monitor.start_monitoring()
            
        elif cmd == '--status':
            # 显示状态
            status = trainer.get_training_status()
            print("📊 训练状态:")
            for modality, info in status.items():
                status_icon = "🔄" if info['running'] else "✅"
                print(f"  {status_icon} {modality.upper()}: {info['status']} (PID: {info['pid']})")
                
        elif cmd == '--stop':
            # 停止训练
            if len(sys.argv) > 2:
                trainer.stop_training(sys.argv[2])
            else:
                trainer.stop_training()
                
        elif cmd in ['smiles', 'graph', 'image']:
            # 启动单个模态
            trainer.start_single_training(cmd)
            
        else:
            print(f"❌ 未知命令: {cmd}")
    
    else:
        # 交互模式
        print("🚀 后台训练启动器")
        print("=" * 50)
        
        print("\n选择训练方式:")
        print("1. 闪电验证 - SMILES模态后台训练 (45分钟)")
        print("2. 完整训练 - 三个模态顺序后台训练 (3小时)")
        print("3. 自定义 - 选择特定模态后台训练")
        print("4. 查看当前训练状态")
        print("5. 停止所有训练")
        
        choice = input("\n请选择 (1-5): ").strip()
        
        if choice == '1':
            trainer.start_single_training('smiles')
            
        elif choice == '2':
            trainer.start_sequential_training()
            
        elif choice == '3':
            modality = input("选择模态 (smiles/graph/image): ").strip().lower()
            if modality in ['smiles', 'graph', 'image']:
                trainer.start_single_training(modality)
            else:
                print("❌ 无效模态!")
                
        elif choice == '4':
            status = trainer.get_training_status()
            if status:
                print("\n📊 当前训练状态:")
                for modality, info in status.items():
                    status_icon = "🔄" if info['running'] else "✅"
                    print(f"  {status_icon} {modality.upper()}: {info['status']} (PID: {info['pid']})")
            else:
                print("📭 没有运行中的训练任务")
                
        elif choice == '5':
            trainer.stop_training()
            
        else:
            print("❌ 无效选择!")

if __name__ == "__main__":
    main()