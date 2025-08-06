#!/usr/bin/env python3
"""
训练恢复指南
"""

import os
from pathlib import Path
import datetime

def check_existing_checkpoints():
    """检查现有的checkpoint文件"""
    checkpoint_dir = Path("/root/autodl-tmp/text2Mol-outputs/safe_training/image")
    
    print("📁 检查现有checkpoint文件...")
    print("=" * 60)
    
    if not checkpoint_dir.exists():
        print("❌ checkpoint目录不存在")
        return []
    
    checkpoints = []
    for file in checkpoint_dir.glob("*.pt"):
        stat = file.stat()
        checkpoints.append({
            'name': file.name,
            'path': str(file),
            'size_gb': stat.st_size / (1024**3),
            'mtime': datetime.datetime.fromtimestamp(stat.st_mtime)
        })
    
    return sorted(checkpoints, key=lambda x: x['mtime'], reverse=True)

def print_resume_guide():
    """打印恢复训练指南"""
    print("\n📚 训练恢复说明")
    print("=" * 60)
    
    checkpoints = check_existing_checkpoints()
    
    if checkpoints:
        print(f"\n✅ 找到 {len(checkpoints)} 个checkpoint文件:")
        for i, ckpt in enumerate(checkpoints):
            print(f"{i+1}. {ckpt['name']} ({ckpt['size_gb']:.1f}GB) - {ckpt['mtime'].strftime('%H:%M:%S')}")
        
        # 检查最新的checkpoint是否完整
        latest = checkpoints[0]
        if latest['size_gb'] < 5.0:  # 正常的checkpoint应该是5.6GB
            print(f"\n⚠️ 最新的checkpoint可能不完整: {latest['name']} 只有 {latest['size_gb']:.1f}GB")
            print("建议使用之前的完整checkpoint")
        
        # 推荐使用的checkpoint
        valid_checkpoints = [c for c in checkpoints if c['size_gb'] >= 5.0]
        if valid_checkpoints:
            recommended = valid_checkpoints[0]
            print(f"\n🎯 推荐使用: {recommended['name']}")
            print(f"   大小: {recommended['size_gb']:.1f}GB")
            print(f"   时间: {recommended['mtime'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n🔧 训练恢复方式:")
    print("\n1. 使用标准train_multimodal.py恢复:")
    print("   python train_multimodal.py \\")
    print("     --scaffold-modality image \\")
    print("     --resume /root/autodl-tmp/text2Mol-outputs/safe_training/image/checkpoint_step_4000.pt \\")
    print("     --output-dir /root/autodl-tmp/text2Mol-outputs/safe_training/image \\")
    print("     --batch-size 6")
    
    print("\n2. 使用改进的脚本恢复:")
    print("   需要先修改improved_safe_training.py支持--resume参数")
    print("   然后: python improved_safe_training.py image --resume checkpoint_step_4000.pt")
    
    print("\n📊 训练进度分析:")
    if checkpoints:
        # 分析训练进度
        step_checkpoints = [c for c in checkpoints if 'step' in c['name']]
        if step_checkpoints:
            # 提取步数
            steps = []
            for c in step_checkpoints:
                try:
                    step = int(c['name'].split('_')[-1].replace('.pt', ''))
                    steps.append(step)
                except:
                    pass
            
            if steps:
                max_step = max(steps)
                # 假设每个epoch大约2686个batch
                epoch_progress = (max_step / 1000) / 2.686  
                print(f"\n当前进度: 约 {epoch_progress:.1f} / 5 epochs")
                print(f"已完成步数: {max_step}")
                print(f"剩余训练: 约 {5 - epoch_progress:.1f} epochs")
    
    print("\n💡 重要提示:")
    print("1. 清理磁盘前，请保留要恢复的checkpoint文件")
    print("2. 建议保留checkpoint_step_4000.pt (最后一个完整的)")
    print("3. 恢复训练会从保存的步数继续，不会重新开始")
    print("4. 确保有足够的磁盘空间(至少15GB)")

def create_resume_script():
    """创建恢复训练的脚本"""
    resume_script = '''#!/usr/bin/env python3
"""
恢复Image模态训练
"""

import subprocess
import sys
import os

# 检查checkpoint文件
checkpoint_path = "/root/autodl-tmp/text2Mol-outputs/safe_training/image/checkpoint_step_4000.pt"

if not os.path.exists(checkpoint_path):
    print(f"❌ Checkpoint文件不存在: {checkpoint_path}")
    print("请先运行: python resume_training_guide.py 查看可用的checkpoint")
    sys.exit(1)

print("🔄 恢复Image模态训练...")
print(f"使用checkpoint: {checkpoint_path}")

# 构建命令
cmd = [
    "python", "train_multimodal.py",
    "--scaffold-modality", "image",
    "--resume", checkpoint_path,
    "--output-dir", "/root/autodl-tmp/text2Mol-outputs/safe_training/image",
    "--batch-size", "6",
    "--epochs", "5",
    "--lr", "1e-5"
]

print(f"\\n执行命令: {' '.join(cmd)}")

# 执行训练
try:
    subprocess.run(cmd, check=True)
except KeyboardInterrupt:
    print("\\n⚠️ 训练被中断")
except Exception as e:
    print(f"\\n❌ 训练出错: {e}")
'''
    
    script_path = Path("resume_image_training.py")
    script_path.write_text(resume_script)
    script_path.chmod(0o755)
    print(f"\n✅ 创建了恢复脚本: {script_path}")

if __name__ == "__main__":
    print_resume_guide()
    create_resume_script()
    
    print("\n" + "=" * 60)
    print("📝 总结：清理后不是完全重新训练！")
    print("- 可以从checkpoint恢复，继续之前的进度")
    print("- 建议保留checkpoint_step_4000.pt")
    print("- 清理其他文件释放空间即可")