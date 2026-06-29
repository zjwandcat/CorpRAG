"""新工具注册模板

本文件是 Agent 平台新工具的注册模板，包含完整的注册流程说明和示例代码。
复制此文件到 ``app/agent/tools/`` 目录下并重命名后，按以下 5 步完成注册。

═══════════════════════════════════════════════════════════════
  工具注册 5 步流程
═══════════════════════════════════════════════════════════════

Step 1 — 复制模板
    将本文件复制到 ``app/agent/tools/`` 目录，并重命名为有意义的名称，
    例如 ``tool_financial_query.py``。

Step 2 — 实现工具逻辑
    在 ``@tool`` 装饰的函数中编写核心业务逻辑。
    - 函数名即工具名（LangChain 约定），请使用 snake_case 命名
    - docstring 是 LLM 决定是否调用此工具的依据，必须清晰描述用途、参数和返回值
    - 参数需带类型标注和默认值（如有）

Step 3 — 在 graph.py 中注册工具
    打开 ``app/agent/graph.py``，在 tools 列表中添加新工具实例::

        from app.agent.tools import my_custom_tool

        tools = [..., my_custom_tool]

Step 4 — 在 auth.py 中配置权限
    打开 ``app/security/auth.py``，在 ``ROLE_PERMISSIONS`` 字典中
    为相应角色添加新工具名称::

        ROLE_PERMISSIONS = {
            "admin": [..., "my_custom_tool"],
            "user": [..., "my_custom_tool"],
        }

Step 5 — 运行测试
    执行单元测试确保工具正常工作::

        pytest tests/unit/test_tools.py -v

═══════════════════════════════════════════════════════════════
"""

from langchain_core.tools import tool


@tool
def my_custom_tool(input_text: str, option: str = "default") -> str:
    """简要描述此工具的用途，LLM 将根据此描述决定是否调用。

    更详细的工具说明可以写在这里，包括使用场景、注意事项等。

    Args:
        input_text: 输入文本，描述第一个参数的含义
        option: 可选参数，描述其含义和可选值，默认为 "default"

    Returns:
        工具执行结果的字符串描述
    """
    # TODO: 在此实现工具的核心业务逻辑
    result = f"处理完成: {input_text} (option={option})"
    return result
