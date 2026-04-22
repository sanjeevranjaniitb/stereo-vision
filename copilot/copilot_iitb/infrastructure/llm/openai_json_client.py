from __future__ import annotations

import asyncio
import json
from typing import Any

from openai import APIConnectionError, APIError, APITimeoutError, AsyncAzureOpenAI, AsyncOpenAI, RateLimitError

from copilot_iitb.config.settings import Settings


def build_async_openai_client(settings: Settings) -> tuple[AsyncOpenAI | AsyncAzureOpenAI, str]:
    """Return (client, chat_model_or_deployment_name)."""
    if settings.llm_provider == "azure_openai":
        return (
            AsyncAzureOpenAI(
                azure_endpoint=settings.azure_openai_endpoint,
                api_version=settings.azure_openai_api_version,
                api_key=settings.azure_openai_api_key,
            ),
            settings.azure_openai_chat_deployment or "",
        )
    if not settings.openai_api_key:
        raise ValueError("OpenAI API key is required for this operation when LLM_PROVIDER=openai.")
    return AsyncOpenAI(api_key=settings.openai_api_key), settings.openai_model


def _is_retriable_openai_error(exc: BaseException) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(exc, APIError):
        code = getattr(exc, "status_code", None)
        return code in (429, 500, 502, 503, 504)
    return False


async def achat_json_object(
    settings: Settings,
    *,
    system: str,
    user: str,
    temperature: float,
) -> dict[str, Any]:
    """Chat completion with JSON response; bounded retries on transient failures."""
    attempts = settings.llm_max_retries + 1
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            client, model = build_async_openai_client(settings)
            resp = await client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                timeout=settings.request_timeout_seconds,
            )
            raw = resp.choices[0].message.content or "{}"
            return json.loads(raw)
        except json.JSONDecodeError as e:
            last = e
        except (APIConnectionError, APITimeoutError, RateLimitError) as e:
            last = e
        except APIError as e:
            if _is_retriable_openai_error(e):
                last = e
            else:
                raise
        if attempt + 1 >= attempts:
            break
        delay = settings.llm_retry_backoff_base_seconds * (2**attempt)
        await asyncio.sleep(delay)
    assert last is not None
    raise last
