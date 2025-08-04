#!/usr/bin/env python3
"""
静默版本的评估脚本，减少RDKit警告输出
"""

import os
import sys
import warnings
import logging
from rdkit import RDLogger

# 禁用RDKit的警告信息
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

# 降低日志级别
logging.getLogger('scaffold_mol_gen').setLevel(logging.ERROR)

# 导入原始评估脚本
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from run_your_model_evaluation import main

if __name__ == '__main__':
    print("🤫 运行静默评估模式（减少RDKit警告）...")
    main()