from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class IEmbeddingProvider(ABC):
    """Embeddings with runtime dimension (no hard-coded vector size in callers)."""

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        raise NotImplementedError

    @abstractmethod
    async def aembed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError
