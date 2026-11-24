"""检查点快照一致性分析。

判定规则（检查点内每个张量的每次读取单独判定）：

1. 读取区间 [r.t_start, r.t_end) 与任意同张量写入区间
   [w.t_start, w.t_end) 真正重叠 => 部分写入（PARTIAL_WRITE），
   该张量读到的是写了一半的数据，快照不可提交。
2. 没有重叠时，读取到的版本 = 读取开始前最后一个完成的写入版本；
   此前没有任何写入完成则为初始版本 INITIAL_VERSION。
3. 检查点内全部张量都干净、且观测到同一个版本 => 可提交
   （COMMITTABLE），检查点只对应这一个优化器版本。
4. 全部干净但版本不统一 => 混合版本（MIXED_VERSION），
   一部分新参数和一部分旧优化器状态被一起保存。

不可提交的检查点不会推进恢复点；恢复时回滚到最近一个可提交检查点，
并重放其后到目标 step 为止的所有 step。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .events import (
    INITIAL_VERSION,
    CheckpointCommit,
    CheckpointRead,
    EventLog,
    TensorWrite,
)

COMMITTABLE = "committable"
"""快照内所有张量来自同一个已完成的优化器版本，可提交。"""

PARTIAL_WRITE = "partial_write"
"""保存期间有张量正在被写入，读取到部分写入的数据。"""

MIXED_VERSION = "mixed_version"
"""读取均已完成但张量版本不一致，跨了两个优化器版本。"""


@dataclass(frozen=True)
class TensorObservation:
    tensor: str
    torn: bool
    observed_version: Optional[int]
    active_step: Optional[int]
    conflicting_write: Optional[TensorWrite]


@dataclass(frozen=True)
class RecomputeRange:
    restore_checkpoint: Optional[str]
    restore_version: int
    from_step: int
    to_step: int
    step_count: int


@dataclass(frozen=True)
class CheckpointReport:
    checkpoint: str
    target_step: int
    commit_t: int
    observations: List[TensorObservation]
    status: str
    consistent_version: Optional[int]
    recompute: Optional[RecomputeRange]

    @property
    def committable(self):
        return self.status == COMMITTABLE


@dataclass(frozen=True)
class AnalysisResult:
    reports: List[CheckpointReport] = field(default_factory=list)

    @property
    def committable_checkpoints(self):
        # type: () -> List[CheckpointReport]
        return [report for report in self.reports if report.committable]

    @property
    def invalid_checkpoints(self):
        # type: () -> List[CheckpointReport]
        return [report for report in self.reports if not report.committable]

    def report_for(self, checkpoint):
        # type: (str) -> CheckpointReport
        for report in self.reports:
            if report.checkpoint == checkpoint:
                return report
        raise KeyError("未知检查点: %r" % checkpoint)


def _intervals_overlap(a_start, a_end, b_start, b_end):
    # type: (int, int, int, int) -> bool
    """半开区间相交判定，端点相接（a_end == b_start）不算重叠。"""
    return a_start < b_end and b_start < a_end


def _observe(read, writes):
    # type: (CheckpointRead, List[TensorWrite]) -> TensorObservation
    same_tensor = [write for write in writes if write.tensor == read.tensor]

    conflicting = next(
        (
            write
            for write in same_tensor
            if _intervals_overlap(
                read.t_start, read.t_end, write.t_start, write.t_end
            )
        ),
        None,
    )
    if conflicting is not None:
        return TensorObservation(
            tensor=read.tensor,
            torn=True,
            observed_version=None,
            active_step=conflicting.step,
            conflicting_write=conflicting,
        )

    completed = [
        write
        for write in same_tensor
        if write.t_end <= read.t_start
    ]
    if not completed:
        observed_version = INITIAL_VERSION
        active_step = None
    else:
        latest = max(completed, key=lambda write: write.t_end)
        observed_version = latest.new_version
        active_step = latest.step
    return TensorObservation(
        tensor=read.tensor,
        torn=False,
        observed_version=observed_version,
        active_step=active_step,
        conflicting_write=None,
    )


def analyze(log):
    # type: (EventLog) -> AnalysisResult
    """分析事件时间线，输出每个检查点的快照一致性结论。"""
    events = log.events
    writes = [event for event in events if isinstance(event, TensorWrite)]
    commits = [
        event for event in events if isinstance(event, CheckpointCommit)
    ]

    reads_by_checkpoint = {}  # type: Dict[str, List[CheckpointRead]]
    for event in events:
        if isinstance(event, CheckpointRead):
            reads_by_checkpoint.setdefault(event.checkpoint, []).append(event)

    reports = []  # type: List[CheckpointReport]
    last_good_checkpoint = None  # type: Optional[str]
    last_good_version = INITIAL_VERSION

    for commit in commits:
        reads = reads_by_checkpoint.get(commit.checkpoint)
        if not reads:
            raise ValueError(
                "检查点 %r 没有任何 CheckpointRead 事件" % commit.checkpoint
            )

        seen_tensors = set()  # type: set
        observations = []  # type: List[TensorObservation]
        for read in reads:
            if read.tensor in seen_tensors:
                raise ValueError(
                    "检查点 %r 对张量 %r 存在多次读取"
                    % (commit.checkpoint, read.tensor)
                )
            seen_tensors.add(read.tensor)
            observations.append(_observe(read, writes))

        torn = [obs for obs in observations if obs.torn]
        if torn:
            status = PARTIAL_WRITE
            consistent_version = None
        else:
            versions = {
                obs.observed_version for obs in observations
            }  # type: set
            if len(versions) == 1:
                status = COMMITTABLE
                consistent_version = next(iter(versions))
            else:
                status = MIXED_VERSION
                consistent_version = None

        recompute = None  # type: Optional[RecomputeRange]
        if status == COMMITTABLE:
            last_good_checkpoint = commit.checkpoint
            last_good_version = consistent_version
        else:
            from_step = last_good_version + 1
            to_step = commit.target_step
            recompute = RecomputeRange(
                restore_checkpoint=last_good_checkpoint,
                restore_version=last_good_version,
                from_step=from_step,
                to_step=to_step,
                step_count=max(0, to_step - from_step + 1),
            )

        reports.append(
            CheckpointReport(
                checkpoint=commit.checkpoint,
                target_step=commit.target_step,
                commit_t=commit.t,
                observations=observations,
                status=status,
                consistent_version=consistent_version,
                recompute=recompute,
            )
        )

    return AnalysisResult(reports=reports)
