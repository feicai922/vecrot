"""轮廓生成适配层（开发文档 §7.3）。"""

from .backends import (
    PolygonResult,
    VectorizeNotAvailable,
    VectorizerBackend,
    build_vectorizer,
)

__all__ = ["PolygonResult", "VectorizeNotAvailable", "VectorizerBackend", "build_vectorizer"]
