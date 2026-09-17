"""
Provider-agnostic LLM factory. A model id with a "/" (e.g. "qwen/qwen3-235b-a22b")
routes to OpenRouter; anything else routes to Google Gemini. LLM_PROVIDER overrides this.
"""
from __future__ import annotations

import os

import config

LLM_TIMEOUT_SECONDS = 30


def resolve_provider(model: str) -> str:
    if config.LLM_PROVIDER:
        return config.LLM_PROVIDER
    return "openrouter" if "/" in model else "google"


def build_chat_model(
    model: str | None = None,
    *,
    temperature: float = 0.3,
    timeout: int = LLM_TIMEOUT_SECONDS,
    max_retries: int = 2,
):
    model = model or config.AGENT_MODEL
    provider = resolve_provider(model)

    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required to use an OpenRouter model")

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
    )
