# 📤 Git推送指南 - 9模态分子生成系统

## 🚀 快速推送步骤

### 1. 检查当前状态
```bash
# 查看当前分支
git branch

# 查看修改状态
git status

# 查看已有的远程仓库
git remote -v
```

### 2. 添加远程仓库（如果还没有）
```bash
# 添加远程仓库
git remote add origin YOUR_REPOSITORY_URL

# 例如：
# GitHub: git remote add origin https://github.com/username/text2mol-scaffold.git
# GitLab: git remote add origin https://gitlab.com/username/text2mol-scaffold.git
```

### 3. 提交代码
```bash
# 添加所有文件
git add .

# 或选择性添加
git add scaffold_mol_gen/
git add train_*.py
git add test_*.py
git add TRAINING_GUIDE_COMPLETE.md
git add CLAUDE.md
git add requirements.txt

# 提交
git commit -m "feat: 9-modal molecular generation system with scaffold constraints

- Implemented multi-modal encoders (SMILES/Graph/Image)
- Added cross-modal fusion with attention mechanism
- Integrated MolT5 for conditional generation
- Created comprehensive training pipeline
- Fixed device consistency issues
- Added evaluation metrics and testing scripts
- Cleaned up project structure (1.8GB -> 47MB)"
```

### 4. 推送到远程
```bash
# 首次推送
git push -u origin main

# 或推送到特定分支
git push -u origin develop

# 后续推送
git push
```

---

## 📋 推送前检查清单

### ✅ 代码准备
- [x] 删除冗余文件（已清理1.75GB）
- [x] 清理__pycache__目录
- [x] 移除大型数据文件（.pkl文件）
- [x] 保留核心训练和测试脚本

### ✅ 文档完善
- [x] README.md - 项目说明
- [x] TRAINING_GUIDE_COMPLETE.md - 详细训练指南
- [x] CLAUDE.md - Claude AI使用指南
- [x] requirements.txt - 依赖列表

### ✅ 敏感信息检查
- [x] 无硬编码密码
- [x] 无API密钥
- [x] 无个人信息
- [x] 路径使用相对路径

---

## 🗂️ 推送的核心文件结构

```
text2mol-scaffold/
├── scaffold_mol_gen/          # 核心代码库 ✅
│   ├── models/               # 模型实现 ✅
│   ├── data/                 # 数据处理 ✅
│   ├── training/             # 训练组件 ✅
│   └── utils/                # 工具函数 ✅
├── Datasets/                  # 数据集CSV ✅
├── configs/                   # 配置文件 ✅
├── docs/                      # 文档 ✅
├── tests/                     # 测试 ✅
├── train_*.py                 # 训练脚本 ✅
├── test_*.py                  # 测试脚本 ✅
├── requirements.txt           # 依赖 ✅
├── README.md                  # 说明 ✅
├── TRAINING_GUIDE_COMPLETE.md # 训练指南 ✅
└── .gitignore                # Git忽略规则 ✅
```

---

## 💾 大文件处理（Git LFS）

如果需要推送大文件（模型权重等）：

### 安装Git LFS
```bash
# Ubuntu/Debian
sudo apt-get install git-lfs

# 初始化
git lfs install
```

### 追踪大文件
```bash
# 追踪模型文件
git lfs track "*.pth"
git lfs track "*.pkl"
git lfs track "*.h5"

# 添加.gitattributes
git add .gitattributes
```

---

## 🏷️ 版本标签

### 创建版本标签
```bash
# 创建标签
git tag -a v1.0.0 -m "Initial release: 9-modal molecular generation"

# 推送标签
git push origin v1.0.0

# 推送所有标签
git push origin --tags
```

---

## 🌿 分支管理

### 推荐分支结构
```bash
main          # 稳定版本
├── develop   # 开发分支
├── feature/* # 功能分支
└── hotfix/*  # 紧急修复
```

### 创建并推送新分支
```bash
# 创建新分支
git checkout -b feature/improved-generation

# 推送新分支
git push -u origin feature/improved-generation
```

---

## 📝 提交信息规范

### 格式
```
<type>(<scope>): <subject>

<body>

<footer>
```

### 类型
- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码格式
- `refactor`: 重构
- `test`: 测试
- `chore`: 构建/工具

### 示例
```bash
git commit -m "feat(training): add 9-modal training pipeline

- Support SMILES/Graph/Image input modalities
- Implement cross-modal fusion mechanism
- Add comprehensive evaluation metrics
- Fix CUDA device consistency issues

Closes #1"
```

---

## 🔧 常见问题

### 1. 文件太大无法推送
```bash
# 错误: Large files detected
# 解决: 使用Git LFS或从历史中删除大文件
git filter-branch --tree-filter 'rm -f path/to/large/file' HEAD
```

### 2. 推送被拒绝
```bash
# 错误: rejected
# 解决: 先拉取再推送
git pull origin main --rebase
git push
```

### 3. 冲突解决
```bash
# 查看冲突文件
git status

# 解决冲突后
git add .
git commit -m "resolve conflicts"
git push
```

---

## 🔒 安全建议

1. **不要推送**:
   - 训练好的模型权重（.pth文件）
   - 大型数据集（.pkl文件）
   - 个人配置文件
   - API密钥

2. **使用环境变量**:
   ```python
   import os
   API_KEY = os.environ.get('API_KEY')
   ```

3. **检查.gitignore**:
   确保所有敏感文件都在.gitignore中

---

## 📊 推送后验证

### GitHub/GitLab界面检查
1. 文件是否完整上传
2. README显示是否正常
3. 代码高亮是否正确
4. 文件大小是否合理

### 克隆测试
```bash
# 在新目录测试克隆
cd /tmp
git clone YOUR_REPOSITORY_URL test-clone
cd test-clone
python -m pytest tests/
```

---

## 🎯 推送命令汇总

```bash
# 完整推送流程
git status                     # 检查状态
git add .                      # 添加所有文件
git commit -m "feat: ..."      # 提交
git push -u origin main        # 推送

# 查看推送历史
git log --oneline -5
```

---

*准备完成！现在可以安全地推送您的代码到远程仓库。*