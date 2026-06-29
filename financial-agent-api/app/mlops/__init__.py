"""MLOps 模块

提供实验追踪、漂移检测、评估框架和 A/B 测试能力。
"""

from app.mlops.ab_testing import ABTestConfig, ABTestMetric, ABTestRouter
from app.mlops.drift_detector import (
    DriftDetectorStatus,
    DriftResult,
    QueryDriftDetector,
    create_drift_detector,
)
from app.mlops.tracking import LLMExperimentTracker

__version__ = "5.0.0"

__all__ = [
    "ABTestConfig",
    "ABTestMetric",
    "ABTestRouter",
    "DriftDetectorStatus",
    "DriftResult",
    "LLMExperimentTracker",
    "QueryDriftDetector",
    "RAGEvaluator",
    "create_drift_detector",
]