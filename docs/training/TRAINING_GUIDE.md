# 多GPU并行训练指南

## 系统准备完成 ✅

### 当前状态
- **磁盘空间**: 39GB可用 (78%空闲)
- **GPU配置**: 双32GB GPU可用
- **模态状态**:
  - SMILES: 已完成基础训练
  - Graph: 待训练
  - Image: 需要重新训练

### 已创建的工具

1. **磁盘守护进程** (`disk_guardian.py`)
   - 每30秒检查磁盘
   - 80%自动清理，90%紧急清理
   - 每个模态最多保留2个checkpoint

2. **训练监控** (`monitor_training.py`)
   - 实时显示GPU、磁盘、进程状态
   - 每5秒更新

3. **快速清理** (`quick_cleanup.py`)
   - 手动清理多余checkpoint

4. **批量启动** (`start_all_training.sh`)
   - 一键启动所有模态训练

## 启动步骤

### 1. 启动磁盘守护进程（新终端）
```bash
cd /root/text2Mol/scaffold-mol-generation
python disk_guardian.py
```

### 2. 启动所有训练（新终端）
```bash
cd /root/text2Mol/scaffold-mol-generation
./start_all_training.sh
```

### 3. 监控训练（新终端）
```bash
cd /root/text2Mol/scaffold-mol-generation
python monitor_training.py
```

## GPU分配策略

- **GPU 0**: SMILES (batch_size=32) + Image (batch_size=10)
- **GPU 1**: Graph (batch_size=16)

两个32GB GPU可以轻松处理这个配置。

## 自动保护机制

1. **磁盘监控**: 每30秒检查，自动清理
2. **Checkpoint限制**: 每个模态最多2-3个文件
3. **保存频率**: 减少到每2000-3000步
4. **紧急停止**: 磁盘<5GB时自动停止

## 监控命令

```bash
# GPU使用率
nvidia-smi -l 1

# 磁盘状态
watch -n 5 'df -h /root/autodl-tmp'

# 查看日志
tail -f logs/*_training.log

# 进程状态
ps aux | grep train
```

## 故障排除

1. **磁盘满**: disk_guardian.py会自动处理
2. **GPU内存不足**: 减小batch_size
3. **训练停止**: 检查logs/目录下的日志

## 预期训练时间

- SMILES: ~2-3小时
- Graph: ~3-4小时  
- Image: ~4-5小时

总计约10小时完成所有模态训练。