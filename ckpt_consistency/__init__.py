"""异步检查点快照一致性验证器。

本包只对“事件时间线”做纯逻辑分析，不加载真实模型、张量或优化器，
也不启动真实线程。时间戳为整数逻辑时钟，仅表示事件先后关系。
"""
from .events import (
    INITIAL_VERSION,
    CheckpointCommit,
    CheckpointRead,
    EventLog,
    GradComplete,
    OptUpdateEnd,
    OptUpdateStart,
    StepStart,
    TensorWrite,
)
from .analyzer import (
    COMMITTABLE,
    MIXED_VERSION,
    PARTIAL_WRITE,
    AnalysisResult,
    CheckpointReport,
    RecomputeRange,
    TensorObservation,
    analyze,
)
from .reporting import format_report

__all__ = [
    "INITIAL_VERSION",
    "CheckpointCommit",
    "CheckpointRead",
    "EventLog",
    "GradComplete",
    "OptUpdateEnd",
    "OptUpdateStart",
    "StepStart",
    "TensorWrite",
    "COMMITTABLE",
    "MIXED_VERSION",
    "PARTIAL_WRITE",
    "AnalysisResult",
    "CheckpointReport",
    "RecomputeRange",
    "TensorObservation",
    "analyze",
    "format_report",
]
