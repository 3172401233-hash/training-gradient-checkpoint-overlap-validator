"""把 AnalysisResult 渲染为可读文本报告。"""
from typing import List

from .analyzer import (
    COMMITTABLE,
    MIXED_VERSION,
    PARTIAL_WRITE,
    AnalysisResult,
    CheckpointReport,
)

STATUS_LABELS = {
    COMMITTABLE: "可提交",
    MIXED_VERSION: "混合版本",
    PARTIAL_WRITE: "部分写入",
}


def _format_observation(observation):
    if observation.torn:
        write = observation.conflicting_write
        return "  - 张量 %-6s 撕裂读：与 step %s 写入(版本 v%s, 区间[%d,%d))重叠" % (
            observation.tensor,
            observation.active_step,
            write.new_version if write else "?",
            write.t_start if write else -1,
            write.t_end if write else -1,
        )
    return (
        "  - 张量 %-6s 干净读：观测版本 v%s（来自 step %s）"
        % (
            observation.tensor,
            observation.observed_version,
            observation.active_step if observation.active_step is not None
            else "初始状态",
        )
    )


def _format_report_one(report):
    # type: (CheckpointReport) -> List[str]
    lines = [
        "检查点 %s（目标 step=%s，提交时刻 t=%d）=> %s"
        % (
            report.checkpoint,
            report.target_step,
            report.commit_t,
            STATUS_LABELS[report.status],
        )
    ]
    for observation in report.observations:
        lines.append(_format_observation(observation))

    if report.committable:
        lines.append(
            "  快照统一于优化器版本 v%s，可提交，恢复点推进。"
            % report.consistent_version
        )
    elif report.status == MIXED_VERSION:
        versions = sorted(
            {
                obs.observed_version
                for obs in report.observations
                if obs.observed_version is not None
            }
        )
        lines.append(
            "  检查点跨越版本 %s，不对应单一优化器版本，禁止提交。"
            % ", ".join("v%s" % version for version in versions)
        )
    else:
        lines.append("  运行中的参数写入混入快照，禁止提交。")

    if report.recompute is not None:
        recompute = report.recompute
        if recompute.step_count == 0:
            lines.append("  恢复重算范围：空（目标 step 已包含在恢复点内）")
        else:
            restore = (
                recompute.restore_checkpoint
                if recompute.restore_checkpoint is not None
                else "初始状态 v0"
            )
            lines.append(
                "  恢复重算范围：回滚到 %s（v%d），重放 step %d..%d，共 %d 个 step"
                % (
                    restore,
                    recompute.restore_version,
                    recompute.from_step,
                    recompute.to_step,
                    recompute.step_count,
                )
            )
    return lines


def format_report(result):
    # type: (AnalysisResult) -> str
    lines = ["检查点快照一致性报告", "=" * 28]
    for report in result.reports:
        lines.extend(_format_report_one(report))

    good = result.committable_checkpoints
    bad = result.invalid_checkpoints
    lines.append("-" * 28)
    lines.append(
        "汇总：可提交 %d 个（%s），不可提交 %d 个（%s）"
        % (
            len(good),
            ", ".join(report.checkpoint for report in good) or "无",
            len(bad),
            ", ".join(report.checkpoint for report in bad) or "无",
        )
    )
    return "\n".join(lines)
