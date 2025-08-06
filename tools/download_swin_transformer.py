#!/usr/bin/env python3
"""
下载Swin Transformer预训练模型到本地
避免训练时需要网络连接
"""

import os
import sys
import requests
from pathlib import Path
import hashlib
from tqdm import tqdm

def download_file(url, dest_path, chunk_size=8192):
    """下载文件并显示进度条"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as file:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
                    pbar.update(len(chunk))

def download_swin_transformer():
    """下载Swin Transformer模型"""
    print("🚀 开始下载Swin Transformer模型")
    print("=" * 60)
    
    # 创建目标目录
    base_dir = Path("/root/autodl-tmp/pretrained_models/swin_transformer")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Swin Transformer模型信息
    # 使用HuggingFace的模型文件
    model_info = {
        'name': 'swin_base_patch4_window7_224.ms_in22k_ft_in1k',
        'url': 'https://huggingface.co/timm/swin_base_patch4_window7_224.ms_in22k_ft_in1k/resolve/main/model.safetensors',
        'config_url': 'https://huggingface.co/timm/swin_base_patch4_window7_224.ms_in22k_ft_in1k/resolve/main/config.json',
        'size': '约350MB'
    }
    
    print(f"📦 模型: {model_info['name']}")
    print(f"📏 大小: {model_info['size']}")
    print(f"📂 保存位置: {base_dir}")
    print()
    
    # 下载模型权重
    model_path = base_dir / "model.safetensors"
    config_path = base_dir / "config.json"
    
    try:
        # 下载模型文件
        if not model_path.exists():
            print("📥 下载模型权重...")
            download_file(model_info['url'], model_path)
            print("✅ 模型权重下载完成")
        else:
            print("✅ 模型权重已存在，跳过下载")
        
        # 下载配置文件
        if not config_path.exists():
            print("\n📥 下载配置文件...")
            download_file(model_info['config_url'], config_path)
            print("✅ 配置文件下载完成")
        else:
            print("✅ 配置文件已存在，跳过下载")
        
        # 创建模型信息文件
        info_path = base_dir / "model_info.txt"
        with open(info_path, 'w') as f:
            f.write(f"Model: {model_info['name']}\n")
            f.write(f"Downloaded from: HuggingFace\n")
            f.write(f"Date: {os.popen('date').read()}")
            f.write("\nUsage: Set environment variable TIMM_MODEL_DIR=/root/autodl-tmp/pretrained_models\n")
        
        print("\n✅ 下载成功!")
        print(f"📁 模型保存在: {base_dir}")
        
        # 提供使用说明
        print("\n📚 使用方法:")
        print("1. 设置环境变量:")
        print("   export TIMM_MODEL_DIR=/root/autodl-tmp/pretrained_models")
        print("\n2. 或在代码中设置:")
        print("   os.environ['TIMM_MODEL_DIR'] = '/root/autodl-tmp/pretrained_models'")
        print("\n3. 模型将自动从本地加载，无需网络连接")
        
    except Exception as e:
        print(f"❌ 下载失败: {str(e)}")
        print("\n💡 备选方案:")
        print("1. 使用代理或VPN")
        print("2. 手动下载并上传")
        print("3. 使用镜像源")
        return False
    
    return True

def create_offline_loader():
    """创建离线加载器脚本"""
    print("\n📝 创建离线加载器...")
    
    loader_script = '''#!/usr/bin/env python3
"""
Swin Transformer离线加载器
修改image_encoder.py以支持本地模型加载
"""

import os
from pathlib import Path

def setup_offline_swin():
    """设置离线Swin Transformer加载"""
    # 设置环境变量
    os.environ['TIMM_MODEL_DIR'] = '/root/autodl-tmp/pretrained_models'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    
    print("✅ 已设置离线模式环境变量")
    print(f"   TIMM_MODEL_DIR: {os.environ['TIMM_MODEL_DIR']}")
    print(f"   HF_HUB_OFFLINE: {os.environ['HF_HUB_OFFLINE']}")

def patch_image_encoder():
    """修补image_encoder.py以支持离线加载"""
    encoder_path = Path("scaffold_mol_gen/models/encoders/image_encoder.py")
    
    if not encoder_path.exists():
        print("❌ 找不到image_encoder.py")
        return
    
    content = encoder_path.read_text()
    
    # 在文件开头添加离线设置
    if "TIMM_MODEL_DIR" not in content:
        import_section = """import logging
import os

# 设置离线模式
os.environ['TIMM_MODEL_DIR'] = '/root/autodl-tmp/pretrained_models'
os.environ['HF_HUB_OFFLINE'] = '1'

"""
        
        # 在导入语句后添加
        import_pos = content.find("import logging")
        if import_pos >= 0:
            content = content[:import_pos] + import_section + content[import_pos+14:]
            encoder_path.write_text(content)
            print("✅ 已修补image_encoder.py支持离线加载")
        else:
            print("⚠️ 无法自动修补，请手动添加环境变量设置")

if __name__ == "__main__":
    setup_offline_swin()
    patch_image_encoder()
'''
    
    loader_path = Path("setup_offline_swin.py")
    loader_path.write_text(loader_script)
    loader_path.chmod(0o755)
    print("✅ 创建了离线加载器: setup_offline_swin.py")

def create_download_alternatives():
    """创建备选下载方案"""
    print("\n📋 创建备选下载方案...")
    
    alternatives = '''#!/bin/bash
# Swin Transformer模型下载备选方案

echo "🔄 Swin Transformer下载备选方案"
echo "================================"

# 方案1: 使用wget下载
echo "方案1: 使用wget直接下载"
mkdir -p /root/autodl-tmp/pretrained_models/swin_transformer
cd /root/autodl-tmp/pretrained_models/swin_transformer

# 尝试不同的下载源
echo "尝试HuggingFace主站..."
wget -c https://huggingface.co/timm/swin_base_patch4_window7_224.ms_in22k_ft_in1k/resolve/main/model.safetensors

# 如果失败，尝试镜像
if [ $? -ne 0 ]; then
    echo "尝试镜像站点..."
    # 这里可以添加其他镜像URL
fi

# 方案2: 使用aria2c多线程下载
echo -e "\\n方案2: 使用aria2c多线程下载"
# aria2c -x 16 -s 16 https://huggingface.co/timm/swin_base_patch4_window7_224.ms_in22k_ft_in1k/resolve/main/model.safetensors

# 方案3: 使用Python的huggingface_hub
echo -e "\\n方案3: 使用huggingface_hub库"
cat > download_with_hf.py << 'EOF'
from huggingface_hub import hf_hub_download

# 下载模型
model_path = hf_hub_download(
    repo_id="timm/swin_base_patch4_window7_224.ms_in22k_ft_in1k",
    filename="model.safetensors",
    cache_dir="/root/autodl-tmp/pretrained_models/",
    resume_download=True
)
print(f"Downloaded to: {model_path}")
EOF

# python download_with_hf.py

echo -e "\\n💡 提示:"
echo "1. 如果下载失败，可以尝试设置代理:"
echo "   export https_proxy=http://your-proxy:port"
echo "2. 或者在其他机器下载后上传到服务器"
echo "3. 模型大小约350MB"
'''
    
    alt_path = Path("download_alternatives.sh")
    alt_path.write_text(alternatives)
    alt_path.chmod(0o755)
    print("✅ 创建了备选方案脚本: download_alternatives.sh")

def main():
    """主函数"""
    print("🌐 Swin Transformer模型下载工具")
    print("=" * 60)
    print("目的: 下载模型到本地，避免训练时的网络依赖")
    print()
    
    # 执行下载
    success = download_swin_transformer()
    
    # 创建辅助脚本
    create_offline_loader()
    create_download_alternatives()
    
    if success:
        print("\n🎉 下载完成!")
        print("下一步:")
        print("1. 运行: python setup_offline_swin.py")
        print("2. 重新启动Graph模态训练")
        print("3. 无需网络即可训练!")
    else:
        print("\n⚠️ 下载失败，请尝试:")
        print("1. 运行: ./download_alternatives.sh")
        print("2. 使用代理或VPN")
        print("3. 手动下载模型文件")

if __name__ == "__main__":
    main()