"""Agent Platform SDK

企业级 GenAI Agent 平台的 Python SDK，提供同步和异步两种客户端，
用于快速集成对话、文档上传、代码审查等平台能力。

示例::

    from sdk import AgentPlatformClient

    with AgentPlatformClient(base_url="http://localhost:8000", api_key="sk-xxx") as client:
        result = client.chat("请帮我查一下报销流程")
        print(result["answer"])
"""

from sdk.agent_platform_client import AgentPlatformClient, AsyncAgentPlatformClient

__all__ = [
    "AgentPlatformClient",
    "AsyncAgentPlatformClient",
]
