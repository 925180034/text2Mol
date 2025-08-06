#!/usr/bin/env python3
"""
Project Cleanup Script
系统清理过期文件、冗余文件，优化项目结构
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime
import re

class ProjectCleaner:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.cleanup_log = []
        self.dry_run = False
        
        # 确保在正确的目录
        if not (self.base_dir / "scaffold_mol_gen").exists():
            raise ValueError("Invalid project directory")
    
    def log_action(self, action, path, reason=""):
        """记录清理操作"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "path": str(path),
            "reason": reason
        }
        self.cleanup_log.append(entry)
        print(f"📁 {action.upper()}: {path} {f'({reason})' if reason else ''}")
    
    def safe_remove(self, path, reason=""):
        """安全删除文件或目录"""
        path = Path(path)
        if not path.exists():
            return
        
        if self.dry_run:
            self.log_action("DRY_RUN_DELETE", path, reason)
            return
        
        try:
            if path.is_file():
                path.unlink()
                self.log_action("DELETED_FILE", path, reason)
            elif path.is_dir():
                shutil.rmtree(path)
                self.log_action("DELETED_DIR", path, reason)
        except Exception as e:
            self.log_action("ERROR", path, f"删除失败: {e}")
    
    def safe_move(self, src, dst, reason=""):
        """安全移动文件或目录"""
        src, dst = Path(src), Path(dst)
        if not src.exists():
            return
        
        if self.dry_run:
            self.log_action("DRY_RUN_MOVE", f"{src} -> {dst}", reason)
            return
        
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            self.log_action("MOVED", f"{src} -> {dst}", reason)
        except Exception as e:
            self.log_action("ERROR", f"{src} -> {dst}", f"移动失败: {e}")
    
    def cleanup_backup_files(self):
        """清理备份文件"""
        print("\n🗂️ 清理备份文件...")
        
        backup_patterns = ["**/*.backup", "**/*.bak", "**/*.old", "**/*~"]
        
        for pattern in backup_patterns:
            for file_path in self.base_dir.glob(pattern):
                self.safe_remove(file_path, "备份文件")
    
    def cleanup_temporary_files(self):
        """清理临时文件和日志"""
        print("\n🗂️ 清理临时文件...")
        
        # 临时文件模式
        temp_patterns = [
            "**/*.tmp", "**/*.temp", "**/*.log", "**/*.cache",
            "**/__pycache__", "**/*.pyc", "**/*.pyo"
        ]
        
        for pattern in temp_patterns:
            for file_path in self.base_dir.glob(pattern):
                self.safe_remove(file_path, "临时文件")
        
        # 特定的临时文件
        temp_files = [
            "evaluation_output.log",
            "simple_generation_results.json"
        ]
        
        for temp_file in temp_files:
            file_path = self.base_dir / temp_file
            if file_path.exists():
                self.safe_remove(file_path, "临时结果文件")
    
    def cleanup_redundant_evaluation_scripts(self):
        """清理冗余的评估脚本"""
        print("\n🗂️ 清理冗余评估脚本...")
        
        # 评估脚本层次结构（从低到高优先级）
        evaluation_scripts = {
            "multimodal_evaluation.py": "早期多模态评估脚本",
            "nine_modality_evaluation.py": "九模态评估原版",
            "simple_nine_modality_eval.py": "简化评估脚本"
        }
        
        # 保留的核心评估脚本
        keep_scripts = [
            "real_model_evaluation.py",  # 主要评估脚本
            "debug_real_evaluation.py",  # 调试版本
            "full_test_evaluation.py",   # 完整测试
            "nine_modality_evaluation_fixed.py"  # 修复版本
        ]
        
        for script, reason in evaluation_scripts.items():
            script_path = self.base_dir / script
            if script_path.exists():
                # 移到archive目录而不是删除
                archive_dir = self.base_dir / "archive" / "evaluation_scripts"
                self.safe_move(script_path, archive_dir / script, f"归档: {reason}")
    
    def cleanup_redundant_preprocessing_scripts(self):
        """清理冗余的预处理脚本"""
        print("\n🗂️ 清理冗余预处理脚本...")
        
        # 预处理脚本层次
        preprocessing_scripts = {
            "preprocess_complete_data.py": "预处理原版",
            "preprocess_save_multimodal_data.py": "保存多模态数据版本"
        }
        
        # 保留最新修复版本
        keep_scripts = [
            "preprocess_complete_data_fixed.py",
            "process_full_test.py"
        ]
        
        for script, reason in preprocessing_scripts.items():
            script_path = self.base_dir / script
            if script_path.exists():
                archive_dir = self.base_dir / "archive" / "preprocessing_scripts"
                self.safe_move(script_path, archive_dir / script, f"归档: {reason}")
    
    def cleanup_old_visualization_results(self):
        """清理旧的可视化结果"""
        print("\n🗂️ 清理旧的可视化结果...")
        
        viz_dir = self.base_dir / "visualization_results"
        if viz_dir.exists():
            # 移动到archive而不是删除，因为可能还有参考价值
            archive_viz_dir = self.base_dir / "archive" / "old_visualizations"
            self.safe_move(viz_dir, archive_viz_dir, "归档旧的可视化结果")
    
    def organize_configs(self):
        """整理配置文件"""
        print("\n🗂️ 整理配置文件...")
        
        # 将根目录的配置文件移到configs目录
        root_configs = [
            "fast_training_config.yaml",
            "safe_training_config.yaml"
        ]
        
        configs_dir = self.base_dir / "configs"
        configs_dir.mkdir(exist_ok=True)
        
        for config in root_configs:
            config_path = self.base_dir / config
            if config_path.exists():
                self.safe_move(config_path, configs_dir / config, "整理配置文件")
    
    def cleanup_experiment_results(self):
        """清理实验结果，保留重要的"""
        print("\n🗂️ 清理实验结果...")
        
        experiments_dir = self.base_dir / "experiments"
        if not experiments_dir.exists():
            return
        
        # 清理具体的过期实验
        old_experiments = [
            "sample_molecule_image.png",
            "short_term_results"
        ]
        
        for exp in old_experiments:
            exp_path = experiments_dir / exp
            if exp_path.exists():
                archive_exp_dir = self.base_dir / "archive" / "old_experiments"
                self.safe_move(exp_path, archive_exp_dir / exp, "归档旧实验结果")
    
    def cleanup_tools_directory(self):
        """清理tools目录中的过期工具"""
        print("\n🗂️ 清理工具目录...")
        
        tools_dir = self.base_dir / "tools"
        if not tools_dir.exists():
            return
        
        # 过期的工具脚本
        deprecated_tools = [
            "emergency_cleanup.py",
            "emergency_cleanup_and_train.py",
            "disk_cleanup_report.py",
            "test_import.py"
        ]
        
        archive_tools_dir = self.base_dir / "archive" / "deprecated_tools"
        
        for tool in deprecated_tools:
            tool_path = tools_dir / tool
            if tool_path.exists():
                self.safe_move(tool_path, archive_tools_dir / tool, "归档过期工具")
    
    def create_archive_structure(self):
        """创建归档目录结构"""
        print("\n📁 创建归档目录结构...")
        
        archive_dirs = [
            "archive/evaluation_scripts",
            "archive/preprocessing_scripts", 
            "archive/old_visualizations",
            "archive/old_experiments",
            "archive/deprecated_tools"
        ]
        
        for dir_path in archive_dirs:
            full_path = self.base_dir / dir_path
            if not self.dry_run:
                full_path.mkdir(parents=True, exist_ok=True)
            self.log_action("CREATED_DIR", full_path, "归档目录")
    
    def generate_cleanup_report(self):
        """生成清理报告"""
        print("\n📊 生成清理报告...")
        
        report = {
            "cleanup_date": datetime.now().isoformat(),
            "project_path": str(self.base_dir),
            "dry_run": self.dry_run,
            "summary": {
                "total_actions": len(self.cleanup_log),
                "files_deleted": len([x for x in self.cleanup_log if x["action"] == "DELETED_FILE"]),
                "dirs_deleted": len([x for x in self.cleanup_log if x["action"] == "DELETED_DIR"]),
                "files_moved": len([x for x in self.cleanup_log if x["action"] == "MOVED"]),
                "dirs_created": len([x for x in self.cleanup_log if x["action"] == "CREATED_DIR"]),
                "errors": len([x for x in self.cleanup_log if x["action"] == "ERROR"])
            },
            "actions": self.cleanup_log
        }
        
        report_file = self.base_dir / "project_cleanup_report.json"
        if not self.dry_run:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.log_action("CREATED_FILE", report_file, "清理报告")
        
        return report
    
    def run_cleanup(self, dry_run=False):
        """执行完整清理"""
        self.dry_run = dry_run
        mode = "预览模式" if dry_run else "执行模式"
        
        print(f"🧹 开始项目清理 ({mode})")
        print("=" * 60)
        
        # 创建归档结构
        self.create_archive_structure()
        
        # 执行各种清理操作
        self.cleanup_backup_files()
        self.cleanup_temporary_files()
        self.cleanup_redundant_evaluation_scripts()
        self.cleanup_redundant_preprocessing_scripts()
        self.cleanup_old_visualization_results()
        self.organize_configs()
        self.cleanup_experiment_results()
        self.cleanup_tools_directory()
        
        # 生成报告
        report = self.generate_cleanup_report()
        
        print("\n" + "=" * 60)
        print("✅ 清理完成!")
        print(f"📊 总操作数: {report['summary']['total_actions']}")
        print(f"🗑️ 删除文件: {report['summary']['files_deleted']}")
        print(f"📁 删除目录: {report['summary']['dirs_deleted']}")
        print(f"📦 移动文件: {report['summary']['files_moved']}")
        print(f"❌ 错误数: {report['summary']['errors']}")
        
        if dry_run:
            print("\n⚠️ 这是预览模式，没有实际修改文件")
            print("要执行清理，请运行: python project_cleanup.py --execute")
        
        return report

def main():
    import sys
    
    base_dir = "/root/text2Mol/scaffold-mol-generation"
    cleaner = ProjectCleaner(base_dir)
    
    # 检查是否是执行模式
    execute = "--execute" in sys.argv or "-e" in sys.argv
    
    if not execute:
        print("🔍 运行预览模式，查看将要执行的操作...")
        cleaner.run_cleanup(dry_run=True)
    else:
        print("⚡ 执行清理操作...")
        cleaner.run_cleanup(dry_run=False)

if __name__ == "__main__":
    main()