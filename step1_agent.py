import json
import os
import socket
import sys
import urllib.parse
from pathlib import Path
from typing import Any, Final

from app.core.enums import ModelName, UserCommand
from app.exceptions import ServiceConnectionError, ToolExecutionError
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA


@tool
def get_stock_pe(stock_code: str) -> float:
    """模拟查询指定股票的市盈率（PE）。

    Args:
        stock_code: 股票代码，例如 "600519"。

    Returns:
        模拟的市盈率数值。
    """
    print(f"[工具执行] get_stock_pe 被调用，参数：stock_code={stock_code}")
    return 25.5


@tool
def search_research_report(company_name: str) -> str:
    """模拟查询指定公司的研报摘要。

    Args:
        company_name: 公司名称，例如 "比亚迪"。

    Returns:
        一段模拟的研报摘要文本。
    """
    print(
        f"[工具执行] search_research_report 被调用，参数：company_name={company_name}",
    )
    return (
        f"{company_name} 近期业绩稳健，新能源汽车销量持续增长，"
        '机构普遍给予"买入"评级，目标价区间上调 10%-15%。'
    )


TOOLS_BY_NAME: Final[dict[str, Any]] = {
    get_stock_pe.name: get_stock_pe,
    search_research_report.name: search_research_report,
}


def _load_env_from_config(filename: str = "nim_config.txt") -> None:
    config_path = Path(__file__).parent / filename
    if not config_path.exists():
        return

    with open(config_path, encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'`")
            if key and not os.getenv(key):
                os.environ[key] = value


def _check_nim_reachable(base_url: str, timeout: float = 3.0) -> bool:
    parsed = urllib.parse.urlparse(base_url)
    host = parsed.hostname
    match parsed:
        case _ if parsed.port:
            port = parsed.port
        case _ if parsed.scheme == "https":
            port = 443
        case _:
            port = 80

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError as exc:
        error = ServiceConnectionError(
            message="无法连接到 NIM 服务",
            service_name="NIM",
            endpoint=base_url,
            details=str(exc),
        )
        print(f"[错误] {error}")
        print("[提示] 请确认 Docker 容器已启动，且端口 8000 已映射到本机。")
        print(
            "[提示] 若 NIM 运行在其它地址，请设置环境变量 NIM_BASE_URL。",
        )
        return False


def create_llm_with_tools() -> Any:
    _load_env_from_config()

    if not os.getenv("NVIDIA_API_KEY"):
        os.environ["NVIDIA_API_KEY"] = "not-needed-for-local-nim"

    base_url = os.getenv("NIM_BASE_URL", "http://localhost:8000/v1").strip().strip("`")
    model_name = os.getenv("NIM_MODEL_NAME", ModelName.DEEPSEEK_V4).strip().strip("`")

    print(f"[配置] 连接 NIM：{base_url}，模型：{model_name}")

    if not _check_nim_reachable(base_url):
        sys.exit(1)

    llm = ChatNVIDIA(
        model=model_name,
        base_url=base_url,
        temperature=1,
        top_p=0.95,
        max_tokens=16384,
    )
    return llm.bind_tools(list(TOOLS_BY_NAME.values()))


def process_turn(
    llm_with_tools: Any,
    messages: list[Any],
    user_input: str,
) -> list[Any]:
    messages.append(HumanMessage(content=user_input))

    ai_message = llm_with_tools.invoke(messages)
    messages.append(ai_message)

    print(f"\n[AI 首轮回复] {ai_message.content or '(无文本，准备调用工具)'}")

    if not ai_message.tool_calls:
        print("[流程] 模型未触发工具调用，直接输出回答。\n")
        return messages

    print("[流程] 模型决定调用以下工具：")
    for tool_call in ai_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        print(f"  - 工具名称：{tool_name}")
        print(f"  - 生成参数：{json.dumps(tool_args, ensure_ascii=False)}")

        tool_func = TOOLS_BY_NAME.get(tool_name)
        match tool_func:
            case None:
                result = str(
                    ToolExecutionError(
                        message="找不到工具",
                        details=f"工具名称: {tool_name}",
                    )
                )
            case _:
                result = tool_func.invoke(tool_args)

        print(f"  - 工具返回：{result}")

        messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call_id,
            ),
        )

    final_message = llm_with_tools.invoke(messages)
    messages.append(final_message)

    print(f"\n[AI 最终回答] {final_message.content}\n")
    return messages


def main() -> None:
    llm_with_tools = create_llm_with_tools()

    system_message = SystemMessage(
        content=(
            "你是金融助手。当用户查询股票市盈率或公司研报时，"
            "你必须先调用对应工具获取信息，再基于工具返回结果给出自然语言回答。"
            "不要在没有调用工具的情况下编造数据。",
        ),
    )
    messages: list[Any] = [system_message]

    print("=" * 60)
    print("LangChain Function Calling 交互示例")
    print("输入问题即可（例如：查一下 600519 的市盈率 / 查一下比亚迪的研报）")
    print("输入 exit / quit 退出")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        match user_input.lower():
            case UserCommand.EXIT | UserCommand.QUIT | UserCommand.EXIT_CN:
                print("再见！")
                break
            case _:
                messages = process_turn(llm_with_tools, messages, user_input)


__all__ = [
    "TOOLS_BY_NAME",
    "create_llm_with_tools",
    "get_stock_pe",
    "main",
    "process_turn",
    "search_research_report",
]


if __name__ == "__main__":
    main()
