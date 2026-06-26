"""协议接口模块，定义核心组件的类型协议。"""

from typing import Any, Protocol


class LLMInterface(Protocol):
    """大语言模型接口协议。"""

    def invoke(self, messages: list[Any]) -> Any: ...
    def bind_tools(self, tools: list[Any]) -> Any: ...


class VectorStoreInterface(Protocol):
    """向量存储接口协议。"""

    def similarity_search(
        self,
        query: str,
        k: int = 3,
        **kwargs: Any,
    ) -> list[Any]: ...

    def add_documents(
        self,
        documents: list[Any],
        **kwargs: Any,
    ) -> list[str]: ...

    def from_documents(
        self,
        documents: list[Any],
        embedding: Any,
        **kwargs: Any,
    ) -> VectorStoreInterface: ...


class EmbeddingsInterface(Protocol):
    """嵌入模型接口协议。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class ToolInterface(Protocol):
    """工具接口协议。"""

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    def invoke(self, tool_input: dict[str, Any]) -> Any: ...


__all__ = [
    "EmbeddingsInterface",
    "LLMInterface",
    "ToolInterface",
    "VectorStoreInterface",
]
