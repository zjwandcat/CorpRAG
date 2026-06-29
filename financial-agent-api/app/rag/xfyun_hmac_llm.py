"""讯飞星辰 HMAC 签名认证适配器

讯飞星辰 MaaS 平台的 /v2/ 端点需要 HMAC 签名认证，
不支持标准 Bearer Token。本模块继承 BaseChatModel，
在每次请求时自动生成 HMAC 签名，对 LangChain/LangGraph 透明。

APIKey 格式：apikey.apisecret（在讯飞星辰控制台获取）
"""

import asyncio
import base64
import hashlib
import hmac
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


def _messages_to_openai(messages: list[BaseMessage]) -> list[dict[str, Any]]:
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


def _parse_openai_response(data: dict[str, Any]) -> AIMessage:
    choices = data.get("choices", [])
    if not choices:
        return AIMessage(content="")

    choice = choices[0]
    message = choice.get("message", {})
    content = message.get("content") or ""

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


def _build_hmac_headers(
    api_key: str,
    api_secret: str,
    host: str,
    path: str,
    method: str = "POST",
) -> dict[str, str]:
    date = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime())
    signature_origin = f"host: {host}\ndate: {date}\n{method} {path} HTTP/1.1"
    signature = base64.b64encode(
        hmac.new(
            api_secret.encode(), signature_origin.encode(), hashlib.sha256
        ).digest()
    ).decode()
    authorization = (
        f'hmac username="{api_key}", algorithm="hmac-sha256", '
        f'headers="host date request-line", signature="{signature}"'
    )
    return {
        "Content-Type": "application/json",
        "Authorization": authorization,
        "Date": date,
        "Host": host,
    }


class XfyunHmacChatModel(BaseChatModel):
    model: str = "xopqwen36v35b"
    api_key: str = ""
    base_url: str = "https://maas-api.cn-huabei-1.xf-yun.com/v2"
    temperature: float = 1.0
    top_p: float = 0.95
    max_tokens: int = 16384
    request_timeout: int = 300
    cache: bool = False

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "xfyun-hmac-chat"

    def _parse_api_key(self) -> tuple[str, str]:
        parts = self.api_key.split(".")
        if len(parts) != 2:
            raise ValueError(
                "讯飞星辰 API Key 格式错误，应为 apikey.apisecret 格式"
            )
        return parts[0], parts[1]

    def _get_host_and_path(self) -> tuple[str, str]:
        from urllib.parse import urlparse
        parsed = urlparse(self.base_url)
        host = parsed.hostname
        path_prefix = parsed.path.rstrip("/")
        return host, path_prefix

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

        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        ai_msg = await self._async_invoke(messages, stop=stop, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    async def _async_invoke(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        apikey, apisecret = self._parse_api_key()
        host, path_prefix = self._get_host_and_path()
        chat_path = f"{path_prefix}/chat/completions"

        openai_messages = _messages_to_openai(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if stop:
            payload["stop"] = stop
        if kwargs.get("tools"):
            payload["tools"] = kwargs["tools"]
        if kwargs.get("tool_choice"):
            payload["tool_choice"] = kwargs["tool_choice"]

        headers = _build_hmac_headers(apikey, apisecret, host, chat_path)

        url = f"https://{host}{chat_path}"

        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            try:
                resp = await client.post(url, json=payload, headers=headers)
            except httpx.TimeoutException as exc:
                raise RuntimeError(f"讯飞星辰请求超时：{exc}") from exc
            except httpx.ConnectError as exc:
                raise RuntimeError(f"讯飞星辰连接失败：{exc}") from exc

            if resp.status_code == 429:
                raise RuntimeError("429 讯飞星辰请求频率超限")

            if resp.status_code == 401:
                err_detail = resp.text[:300]
                raise RuntimeError(
                    f"讯飞星辰认证失败（401）：{err_detail}"
                )

            if resp.status_code != 200:
                err_detail = resp.text[:300]
                raise RuntimeError(
                    f"讯飞星辰请求失败 HTTP {resp.status_code}：{err_detail}"
                )

            data = resp.json()
            return _parse_openai_response(data)