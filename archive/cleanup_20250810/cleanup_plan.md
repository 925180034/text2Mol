# 清理计划

## 待清理文件列表

### 1. 冗余的测试脚本
- test_multimodal_simple.py (被run_fixed_multimodal_test.py替代)
- test_simple_cases.py (临时测试)
- evaluate_multimodal_comprehensive.py (旧版本)
- evaluate_fixed_multimodal.py (已被整合)
- run_all_multimodal_test.py (被run_fixed_multimodal_test.py替代)

### 2. 临时修复脚本（已集成到主代码）
- fix_generation_quality.py (已集成)
- fix_graph_input.py (已集成)
- fix_image_input.py (已集成)
- test_all_fixes.py (临时测试)

### 3. 实验性训练脚本
- test_with_correct_molt5.py (实验性质)
- test_with_trained_model.py (临时测试)
- train_molt5_for_smiles.py (未完成的实验)

### 4. 冗余报告文件
- MULTIMODAL_STATUS_REPORT.md (过时)
- FINAL_MULTIMODAL_TEST_REPORT.md (过时)
- component_test_*.json (临时测试结果)
- multimodal_capability_test_*.json (临时测试结果)

### 5. __pycache__目录
- 9个__pycache__目录需要清理

## 保留文件

### 核心训练脚本
- train_fixed_multimodal.py
- train_joint_multimodal.py
- train_optimized_32gb.py
- train_scaffold_completion.py

### 核心测试脚本
- run_fully_fixed_test.py
- run_fixed_multimodal_test.py
- run_test.py
- run_multimodal.sh

### 启动脚本
- launch_32gb_training.sh
- launch_production_training.sh
- start_background_training.sh

### 重要文档
- README.md
- FIX_SUMMARY_REPORT.md (保留作为参考)

## 清理策略
1. 将冗余文件移到archive/cleanup_20250810目录
2. 删除所有__pycache__目录
3. 清理test_results中的旧文件
4. 整理主目录结构