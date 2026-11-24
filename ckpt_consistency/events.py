"""训练与检查点事件模型。

模拟的时间线包含七类事件：

* StepStart        一个训练 step 开始（反向传播在该 step 内完成）
* GradComplete     该 step 梯度计算完成
* OptUpdateStart   优化器开始把新版本写入参数与优化器状态
* TensorWrite      单个张量的一次非原子写入（带时间区间）
* OptUpdateEnd     优化器更新全部结束，此时参数版本整体切换
* CheckpointRead   检查点保存线程对单个张量的读取（带时间区间）
* CheckpointCommit 检查点保存完成、尝试提交

所有时间戳都是整数逻辑时钟。区间采用半开语义 [t_start, t_end)：
读取区间与写入区间仅在端点相接（read.t_start == write.t_end）时不算
重叠，因为该写入在读取开始前已经完成。
"""
from dataclasses import dataclass
from typing import Any, List

INITIAL_VERSION = 0
"""第 1 个 step 开始前，所有参数与优化器状态的统一版本。"""


@dataclass(frozen=True)
class StepStart:
    t: int
    step: int


@dataclass(frozen=True)
class GradComplete:
    t: int
    step: int


@dataclass(frozen=True)
class OptUpdateStart:
    t: int
    step: int
    new_version: int


@dataclass(frozen=True)
class TensorWrite:
    """优化器更新对单个张量的非原子写入。"""

    t_start: int
    t_end: int
    step: int
    new_version: int
    tensor: str


@dataclass(frozen=True)
class OptUpdateEnd:
    t: int
    step: int
    new_version: int


@dataclass(frozen=True)
class CheckpointRead:
    """保存线程读取一个张量，读取期间该张量可能正被写入。"""

    t_start: int
    t_end: int
    checkpoint: str
    tensor: str


@dataclass(frozen=True)
class CheckpointCommit:
    t: int
    checkpoint: str
    target_step: int


class EventLog:
    """收集事件，并做基本合法性校验。

    点事件（开始、完成、提交）时间戳必须单调不减；区间事件
    （写入、读取）只要求非负且 t_end >= t_start。区间允许互相
    交错，因为“读取在写入尚未结束时开始”正是要验证的竞争本身，
    区间的重叠关系交给 analyzer 按半开语义判定。
    """

    def __init__(self):
        self._events = []  # type: List[Any]
        self._last_point_t = -1

    @property
    def events(self):
        # type: () -> List[Any]
        return list(self._events)

    def _check_point(self, t):
        # type: (int) -> None
        if isinstance(t, bool) or not isinstance(t, int):
            raise TypeError("时间戳必须是 int，得到 %r" % type(t).__name__)
        if t < 0:
            raise ValueError("时间戳不能为负: %d" % t)
        if t < self._last_point_t:
            raise ValueError(
                "点事件时间戳必须单调不减: %d 早于上一点事件时间 %d"
                % (t, self._last_point_t)
            )
        self._last_point_t = t

    def _check_interval(self, t_start, t_end):
        # type: (int, int) -> None
        for value in (t_start, t_end):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    "区间时间戳必须是 int，得到 %r" % type(value).__name__
                )
            if value < 0:
                raise ValueError("时间戳不能为负: %d" % value)
        if t_end < t_start:
            raise ValueError(
                "区间结束时间 %d 早于开始时间 %d" % (t_end, t_start)
            )

    def add(self, event):
        # type: (Any) -> Any
        if isinstance(event, (TensorWrite, CheckpointRead)):
            self._check_interval(event.t_start, event.t_end)
        else:
            self._check_point(event.t)
        self._events.append(event)
        return event

    def step_start(self, t, step):
        return self.add(StepStart(t=t, step=step))

    def grad_complete(self, t, step):
        return self.add(GradComplete(t=t, step=step))

    def opt_update_start(self, t, step, new_version):
        return self.add(
            OptUpdateStart(t=t, step=step, new_version=new_version)
        )

    def tensor_write(self, t_start, t_end, step, new_version, tensor):
        return self.add(
            TensorWrite(
                t_start=t_start,
                t_end=t_end,
                step=step,
                new_version=new_version,
                tensor=tensor,
            )
        )

    def opt_update_end(self, t, step, new_version):
        return self.add(OptUpdateEnd(t=t, step=step, new_version=new_version))

    def checkpoint_read(self, t_start, t_end, checkpoint, tensor):
        return self.add(
            CheckpointRead(
                t_start=t_start,
                t_end=t_end,
                checkpoint=checkpoint,
                tensor=tensor,
            )
        )

    def checkpoint_commit(self, t, checkpoint, target_step):
        return self.add(
            CheckpointCommit(
                t=t, checkpoint=checkpoint, target_step=target_step
            )
        )

    @classmethod
    def from_events(cls, events):
        # type: (List[Any]) -> EventLog
        log = cls()
        for event in events:
            log.add(event)
        return log
