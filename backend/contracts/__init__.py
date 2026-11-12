"""协议契约：schema 版本、事件发射、UTF-8 流解析。"""

from .version import SCHEMA_VERSION, EVENT_TYPES, STAGES, EXIT_CODES, JobState

__all__ = ["SCHEMA_VERSION", "EVENT_TYPES", "STAGES", "EXIT_CODES", "JobState"]
