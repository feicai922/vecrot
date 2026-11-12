"""协议常量与作业状态机（开发文档 §8.3）。

状态迁移集中在这里校验，避免协调器各处随手改状态导致台账与事件不一致。
"""
from __future__ import annotations

import enum

SCHEMA_VERSION = "1.0"

# 事件类别：UI 只依赖这些名字；未识别的事件按兼容性记录，不猜测语义。
EVENT_TYPES = frozenset({
    "ready",
    "stage_started",
    "progress",
    "warning",
    "preview_ready",
    "stage_finished",
    "completed",
    "failed",
    "cancelled",
})

# 阶段名：与 §8.2「阶段分别统计窗口数、候选框数、保留框数、成功轮廓数」对应。
STAGES = ("VALIDATING", "PREPARING", "DETECTING", "DEDUPLICATING",
          "VECTORIZING", "EXPORTING")

# 协调器退出码（§8.3）
EXIT_CODES = {"ok": 0, "partial": 10, "cancelled": 20, "failed": 1}


class JobState(str, enum.Enum):
    CREATED = "CREATED"
    VALIDATING = "VALIDATING"
    PREPARING = "PREPARING"
    DETECTING = "DETECTING"
    DEDUPLICATING = "DEDUPLICATING"
    VECTORIZING = "VECTORIZING"
    EXPORTING = "EXPORTING"
    SUCCEEDED = "SUCCEEDED"
    PARTIAL = "PARTIAL"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


TERMINAL = frozenset({JobState.SUCCEEDED, JobState.PARTIAL,
                      JobState.CANCELLED, JobState.FAILED})

_ACTIVE = (JobState.CREATED, JobState.VALIDATING, JobState.PREPARING,
           JobState.DETECTING, JobState.DEDUPLICATING, JobState.VECTORIZING,
           JobState.EXPORTING, JobState.CANCELLING)

#: 主链路顺序迁移 + 取消/失败边。表里没有的组合一律拒绝。
_ALLOWED = set()
for _a, _b in zip(_ACTIVE[:-1], _ACTIVE[1:]):
    _ALLOWED.add((_a, _b))
for _a in _ACTIVE:
    _ALLOWED.add((_a, JobState.CANCELLING))
    _ALLOWED.add((_a, JobState.FAILED))
_ALLOWED.add((JobState.CANCELLING, JobState.CANCELLED))
_ALLOWED.add((JobState.CANCELLING, JobState.FAILED))
_ALLOWED.add((JobState.EXPORTING, JobState.SUCCEEDED))
_ALLOWED.add((JobState.EXPORTING, JobState.PARTIAL))
# 恢复：已落盘的中间态可重新进入对应阶段
for _s in (JobState.DETECTING, JobState.DEDUPLICATING, JobState.VECTORIZING,
           JobState.EXPORTING, JobState.VALIDATING, JobState.PREPARING):
    _ALLOWED.add((_s, _s))


def can_transition(src: "JobState | str", dst: "JobState | str") -> bool:
    return (JobState(src), JobState(dst)) in _ALLOWED


def check_transition(src: "JobState | str", dst: "JobState | str") -> None:
    if not can_transition(src, dst):
        raise ValueError(f"illegal state transition {JobState(src).value} -> {JobState(dst).value}")


STAGE_STATE = {
    "VALIDATING": JobState.VALIDATING,
    "PREPARING": JobState.PREPARING,
    "DETECTING": JobState.DETECTING,
    "DEDUPLICATING": JobState.DEDUPLICATING,
    "VECTORIZING": JobState.VECTORIZING,
    "EXPORTING": JobState.EXPORTING,
}
