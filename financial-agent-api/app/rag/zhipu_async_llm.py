"""智谱AI 异步对话补全适配器

通过 /paas/v4/async/chat/completions 提交任务，
再通过 /paas/v4/async-result/{id} 轮询结果，
避免同步接口的 429 速率限制问题。

继承 langchain_core.language_models.chat_models.BaseChatModel，
对 LangGraph / LangChain 调用链完全透明。
"""

import asyncio
import json
import logging
import time
from typing import Any

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

logger = logging.getLogger(__name__)

_ASYNC_CHAT_PATH = "/async/chat/completions"
_ASYNC_RESULT_PATH = "/async-result/{task_id}"

_DEFAULT_POLL_INTERVAL = 2.0
_DEFAULT_MAX_POLL_TIME = 300


def _messages_to_zhipu(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.type
        if role == "system":
            role = "system"
        elif role == "human":
            role = "user"
        elif role == "ai":
            role = "assistant"
        elif role == "tool":
            role = "tool"
        entry: dict[str, Any] = {"role": role, "content": msg.content}
        if isinstance(msg, AIMessage) and msg.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": tc.get("id", f"call_{i}"),
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["args"], ensure_ascii=False),
                    },
                }
                for i, tc in enumerate(msg.tool_calls)
            ]
        if role == "tool" and hasattr(msg, "tool_call_id"):
            entry["tool_call_id"] = msg.tool_call_id
        result.append(entry)
    return result


def _parse_zhipu_response(data: dict[str, Any]) -> AIMessage:
    choices = data.get("choices", [])
    if not choices:
        return AIMessage(content="")

    choice = choices[0]
    message = choice.get("message", {})
    content = message.get("content") or ""
    reasoning_content = message.get("reasoning_content") or ""

    tool_calls_raw = message.get("tool_calls") or []
    parsed_tool_calls = []
    for tc in tool_calls_raw:
        fn = tc.get("function", {})
        args_str = fn.get("arguments", "{}")
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except (json.JSONDecodeError, TypeError):
            args = {}
        parsed_tool_calls.append(
            {
                "id": tc.get("id", ""),
                "name": fn.get("name", ""),
                "args": args,
            }
        )

    finish_reason = choice.get("finish_reason", "")

    additional_kwargs: dict[str, Any] = {}
    if reasoning_content:
        additional_kwargs["reasoning_content"] = reasoning_content
    if finish_reason:
        additional_kwargs["finish_reason"] = finish_reason

    usage = data.get("usage")
    if usage:
        additional_kwargs["usage"] = usage

    return AIMessage(
        content=content,
        tool_calls=parsed_tool_calls,
        additional_kwargs=additional_kwargs,
    )


class ZhipuAsyncChatModel(BaseChatModel):
    model: str = "glm-4.7-flash"
    api_key: str = ""
    base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    temperature: float = 1.0
    top_p: float = 0.95
    max_tokens: int = 16384
    request_timeout: int = 300
    poll_interval: float = _DEFAULT_POLL_INTERVAL
    max_poll_time: int = _DEFAULT_MAX_POLL_TIME
    cache: bool = False

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "zhipu-async-chat"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                ai_msg = pool.submit(
                    asyncio.run,
                    self._async_invoke(messages, stop=stop, **kwargs),
                ).result()
        else:
            ai_msg = asyncio.run(self._async_invoke(messages, stop=stop, **kwargs))

        return ChatResult(
            generations=[
                ChatGeneration(message=ai_msg),
            ]
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        ai_msg = await self._async_invoke(messages, stop=stop, **kwargs)
        return ChatResult(
            generations=[
                ChatGeneration(message=ai_msg),
            ]
        )

    async def _async_invoke(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        zhipu_messages = _messages_to_zhipu(messages)

        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": zhipu_messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if stop:
            payload["stop"] = stop
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            submit_url = f"{self.base_url}{_ASYNC_CHAT_PATH}"
            submit_start = time.monotonic()

            try:
                resp = await client.post(submit_url, json=payload, headers=headers)
            except httpx.TimeoutException as exc:
                raise LLMTimeoutError(f"智谱异步提交超时：{exc}") from exc
            except httpx.ConnectError as exc:
                raise LLMConnectionError(f"智谱连接失败：{exc}") from exc

            if resp.status_code == 429:
                raise LLMRateLimitError("智谱异步提交 429 限流")

            if resp.status_code != 200:
                err_detail = resp.text[:300]
                raise LLMInvocationError(
                    f"智谱异步提交失败 HTTP {resp.status_code}：{err_detail}"
                )

            submit_data = resp.json()
            task_id = submit_data.get("id")
            if not task_id:
                raise LLMInvocationError(f"智谱异步提交未返回 task_id：{submit_data}")

            logger.info("智谱异步任务已提交：task_id=%s", task_id)

            poll_deadline = time.monotonic() + self.max_poll_time
            while time.monotonic() < poll_deadline:
                await asyncio.sleep(self.poll_interval)

                result_url = f"{self.base_url}{_ASYNC_RESULT_PATH.format(task_id=task_id)}"
                try:
                    result_resp = await client.get(result_url, headers=headers)
                except httpx.TimeoutException:
                    continue
                except httpx.ConnectError as exc:
                    logger.warning("智谱轮询连接失败，继续重试：%s", exc)
                    continue

                if result_resp.status_code != 200:
                    logger.warning(
                        "智谱轮询 HTTP %d，继续重试", result_resp.status_code
                    )
                    continue

                result_data = result_resp.json()
                task_status = result_data.get("task_status", "")

                if task_status == "SUCCESS":
                    elapsed_ms = int((time.monotonic() - submit_start) * 1000)
                    logger.info(
                        "智谱异步任务完成：task_id=%s, elapsed=%dms", task_id, elapsed_ms
                    )
                    return _parse_zhipu_response(result_data)

                if task_status == "FAIL":
                    error_msg = result_data.get("error", {}).get("message", "未知错误")
                    raise LLMInvocationError(f"智谱异步任务失败：{error_msg}")

            raise LLMTimeoutError(
                f"智谱异步任务轮询超时（{self.max_poll_time}s）：task_id={task_id}"
            )


class LLMTimeoutError(Exception):
    pass


class LLMConnectionError(Exception):
    pass


class LLMRateLimitError(Exception):
    pass


class LLMInvocationError(Exception):
    pass