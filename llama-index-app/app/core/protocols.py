from typing import Any, Protocol


class LLMInterface(Protocol):
    """LlamaIndex LLM 接口协议"""

    def complete(self, prompt: str, **kwargs: Any) -> Any: ...
    def chat(self, messages: list[Any], **kwargs: Any) -> Any: ...
    def achat(self, messages: list[Any], **kwargs: Any) -> Any: ...


class IndexInterface(Protocol):
    """LlamaIndex VectorStoreIndex 接口协议"""

    def as_retriever(self, **kwargs: Any) -> Any: ...
    def as_query_engine(self, **kwargs: Any) -> Any: ...
    def insert(self, document: Any, **kwargs: Any) -> None: ...
    def refresh(self, documents: list[Any], **kwargs: Any) -> list[bool]: ...


class RetrieverInterface(Protocol):
    """LlamaIndex 检索器接口协议"""

    def retrieve(self, query: str, **kwargs: Any) -> list[Any]: ...


class AgentInterface(Protocol):
    """LlamaIndex Agent 接口协议"""

    def chat(self, message: str, **kwargs: Any) -> Any: ...
    def reset(self) -> None: ...


__all__ = [
    "AgentInterface",
    "IndexInterface",
    "LLMInterface",
    "RetrieverInterface",
]
