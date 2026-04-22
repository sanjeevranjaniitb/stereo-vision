from __future__ import annotations

import hashlib
import struct
from typing import List

import numpy as np


class LocalHashEmbeddings:
    """Deterministic pseudo-embeddings for offline dev; dimension is configurable."""

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def _vector_for(self, text: str) -> List[float]:
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        raw = (seed * ((self.dimensions // len(seed)) + 1))[: self.dimensions]
        ints = struct.unpack(f"{len(raw)}B", raw)
        vec = np.array(ints, dtype=np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        return vec.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._vector_for("query:" + text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vector_for("doc:" + t) for t in texts]
