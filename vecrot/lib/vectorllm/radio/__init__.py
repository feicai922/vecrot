"""C-RADIO v2.5 视觉塔（NVIDIA RADIO，Apache-2.0）。

本子包是 `vectorllm_v1` 目录里那份 RADIO 实现的**唯一副本**：
v1 里根目录 22 个 .py 与 `radio_bundle/` 是同一批文件的重复拷贝，
其中 `radio_bundle/` 在运行时从未被 import。v2 只保留一份，放在这里。

对外只暴露两个名字，内部模块之间的相对 import 保持不变：
"""

from .hf_model import RADIOConfig, RADIOModel

__all__ = ["RADIOConfig", "RADIOModel"]
