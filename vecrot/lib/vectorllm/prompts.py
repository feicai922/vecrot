"""VectorLLM 的 prompt 与生成超参。

注意：**必须用 raw chat 模板**（手写 `<|im_start|>`...），不要先 `apply_chat_template`
再拼 `<pixel>` —— `<pixel>` 会在 processor 里展开成 64 个视觉占位 token。
"""

from __future__ import annotations

BUILDING_QUESTION = (
    "Please extract the regular vector contour of the central building in the image, "
    "start from the left top corner and in clockwise."
)
OBJECT_QUESTION = (
    "Please extract the contour of the central object in the image, "
    "start from the left top corner and in clockwise."
)

PIXEL_TOKEN = "<pixel>"
DEFAULT_MAX_NEW_TOKENS = 640  # 每顶点 2 token ⇒ 上限约 320 个顶点
STOP_WORDS = ("<|im_end|>", "<|endoftext|>")


def raw_prompt(question: str = BUILDING_QUESTION) -> str:
    """拼出完整的 raw 对话串（含 assistant 起始标记）。"""
    return (
        f"<|im_start|>user\n{PIXEL_TOKEN}\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


BUILDING_RAW_PROMPT = raw_prompt(BUILDING_QUESTION)
OBJECT_RAW_PROMPT = raw_prompt(OBJECT_QUESTION)


def prompt_for(target: str = "building") -> str:
    target = (target or "building").lower()
    return OBJECT_RAW_PROMPT if target == "object" else BUILDING_RAW_PROMPT
