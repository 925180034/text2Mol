# 项目结构

## 核心代码
- `scaffold_mol_gen/` - 核心模型和训练代码
  - `models/` - 模型架构(编码器、融合层、生成器)
  - `data/` - 数据处理和加载器
  - `training/` - 训练循环和指标
  - `evaluation/` - 评估系统

## 训练和评估
- `train_multimodal.py` - 主训练脚本
- `multimodal_evaluation.py` - 多模态评估脚本
- `test_all_modalities.py` - 模态测试脚本

## 配置和数据
- `configs/` - 训练配置文件
- `Datasets/` - ChEBI-20数据集(train/val/test)

## 工具和测试
- `tools/` - 辅助工具脚本
- `tests/` - 单元测试和集成测试
- `scripts/` - 实用脚本

## 结果和报告
- `evaluation_results/` - 评估结果
- `final_evaluation_report.md` - 最终评估报告
- `experiments/` - 实验记录

## 文档
- `CLAUDE.md` - Claude Code指导文档
- `PROJECT_STATUS.md` - 项目状态
- `MASTER_IMPLEMENTATION_PLAN.md` - 实施计划
- `IMPLEMENTATION_CHECKLIST.md` - 进度清单

## 存档
- `archive/` - 历史代码和参考资料

## 训练模型位置
模型保存在: `/root/autodl-tmp/text2Mol-outputs/fast_training/`
- SMILES: `smiles/final_model.pt`
- Graph: `graph/checkpoint_step_5000.pt`
- Image: `image/checkpoint_step_4000.pt`
