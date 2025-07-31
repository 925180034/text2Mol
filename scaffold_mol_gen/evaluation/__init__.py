"""
Evaluation modules for scaffold-based molecular generation.

This package provides comprehensive evaluation tools for:
- Model performance assessment
- Generation quality metrics
- Comparative analysis
- Benchmark evaluation
"""

from .evaluator import ModelEvaluator, BenchmarkEvaluator

__all__ = [
    'ModelEvaluator',
    'BenchmarkEvaluator'
]