#!/usr/bin/env python3
"""
修复版快速启动脚本
"""

import subprocess
import os
import sys
import time
from pathlib import Path

# 确保在正确的目录
os.chdir('/root/text2Mol/scaffold-mol-generation')

# 创建必要的目录
Path("logs").mkdir(exist_ok=True)
Path("/root/autodl-tmp/text2Mol-outputs/fast_training").mkdir(parents=True, exist_ok=True)

def test_training_script():
    """测试训练脚本是否可用"""
    if not Path("train_multimodal.py").exists():
        print("❌ 错误: train_multimodal.py 不存在!")
        print("当前目录:", os.getcwd())
        print("目录内容:", os.listdir("."))
        return False
    
    # 测试导入
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import sys; sys.path.append('.'); from train_multimodal import *"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            print("⚠️ 训练脚本导入测试失败:")
            print(result.stderr[:500])
            return False
    except Exception as e:
        print(f"⚠️ 测试失败: {e}")
        return False
    
    return True

def start_simple_training(modality, gpu_id):
    """简化的训练启动"""
    
    print(f"\n🚀 启动 {modality} 训练 (GPU {gpu_id})")
    
    # 创建输出目录
    output_dir = f"/root/autodl-tmp/text2Mol-outputs/fast_training/{modality}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 优化的参数
    params = {
        'smiles': {'batch': 32, 'lr': 3e-5},  # 减小batch size避免OOM
        'graph': {'batch': 16, 'lr': 2e-5}, 
        'image': {'batch': 8, 'lr': 1e-5}
    }
    
    config = params[modality]
    
    # 构建命令
    cmd = [
        sys.executable,
        "train_multimodal.py",
        "--scaffold-modality", modality,
        "--output-dir", output_dir,
        "--batch-size", str(config['batch']),
        "--lr", str(config['lr']),
        "--epochs", "1",  # 先训练1个epoch测试
    ]
    
    # 设置环境变量
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # 日志文件
    log_file = f"logs/fast_{modality}_{time.strftime('%H%M%S')}.log"
    print(f"  日志: {log_file}")
    print(f"  命令: {' '.join(cmd)}")
    
    try:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                cwd='/root/text2Mol/scaffold-mol-generation'  # 明确指定工作目录
            )
            print(f"  ✅ PID: {process.pid}")
            return process
    except Exception as e:
        print(f"  ❌ 启动失败: {e}")
        return None

def monitor_training(processes):
    """监控训练进度"""
    print("\n📊 监控训练进度...")
    print("查看日志: tail -f logs/fast_*.log")
    print("查看GPU: nvidia-smi -l 1")
    print("\n等待训练...\n")
    
    start_time = time.time()
    
    # 监控循环
    while any(p and p.poll() is None for p in processes.values()):
        time.sleep(30)
        
        elapsed = time.time() - start_time
        status = f"⏱️ 运行时间: {elapsed/60:.1f}分钟"
        
        for modality, proc in processes.items():
            if proc:
                if proc.poll() is None:
                    status += f" | {modality}: 运行中"
                else:
                    status += f" | {modality}: 完成(返回码:{proc.returncode})"
        
        print(f"\r{status}", end='', flush=True)
    
    print("\n\n✅ 训练结束!")
    
    # 检查结果
    for modality, proc in processes.items():
        if proc and proc.returncode != 0:
            print(f"⚠️ {modality} 训练可能失败，请检查日志")

def main():
    print("🔧 修复版快速训练启动器")
    print("=" * 60)
    
    # 测试环境
    print("\n1. 测试环境...")
    if not test_training_script():
        print("\n请修复训练脚本后重试")
        return
    
    print("✅ 环境测试通过")
    
    # 检查GPU
    print("\n2. 检查GPU...")
    try:
        result = subprocess.run(['nvidia-smi', '--list-gpus'], capture_output=True, text=True)
        print(result.stdout)
    except:
        print("⚠️ 无法检查GPU")
    
    # 启动训练
    print("\n3. 开始启动训练...")
    
    processes = {}
    
    # 先只启动SMILES测试
    processes['smiles'] = start_simple_training('smiles', 0)
    
    if processes['smiles']:
        print("\n等待10秒检查是否正常运行...")
        time.sleep(10)
        
        if processes['smiles'].poll() is None:
            print("✅ SMILES训练正常运行")
            
            # 启动其他模态
            processes['graph'] = start_simple_training('graph', 1)
            
            # 监控所有训练
            monitor_training(processes)
        else:
            print("❌ SMILES训练启动失败")
            print("检查日志: tail logs/fast_smiles_*.log")
    
    print("\n训练启动器结束")

if __name__ == "__main__":
    main()