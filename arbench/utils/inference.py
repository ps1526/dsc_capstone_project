"""
Inference utilities for AR-Bench
Supports OpenAI-compatible APIs (vLLM, OpenAI, Together, etc.)
"""
import os
from typing import Dict, List, Optional

from openai import OpenAI

_DEFAULT_MODEL = "qwen2.5-7b-instruct"
_DEFAULT_TEMPERATURE = 1.0
_DEFAULT_TOP_P = 0.98

def inference(
    messages: List[Dict],
    model: Optional[str] = None,
    json_format: bool = False,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
):
    try:
        model = model or _DEFAULT_MODEL
        temperature = max(temperature or _DEFAULT_TEMPERATURE, 1.0)   # force ≥1.0
        top_p = max(top_p or _DEFAULT_TOP_P, 0.98)

        # Force IPv4
        if base_url and "localhost" in base_url:
            base_url = base_url.replace("localhost", "127.0.0.1")
        base_url = base_url or os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8000/v1")

        print(f"[Inference] base_url={base_url}")
        print(f"[Inference] params: temp={temperature}, top_p={top_p}, model={model}")

        client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
            base_url=base_url,
            timeout=300.0,
            max_retries=5,
        )

        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": 1024,                # hard minimum
            "presence_penalty": 0.4,           # encourage new tokens
            "frequency_penalty": 0.4,
            "repetition_penalty": 1.2,         # stronger anti-repetition
        }

        if json_format:
            kwargs["response_format"] = {"type": "json_object"}

        print("[Inference] Sending request...")
        response = client.chat.completions.create(**kwargs)

        # Debug the actual content
        content = response.choices[0].message.content or ""
        print(f"[Inference] Raw model reply (first 200 chars): {content[:200]}")
        if not content.strip():
            print("[Inference] WARNING: model returned empty string!")

        return response

    except Exception as e:
        print(f"[Inference ERROR] {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

        class MockResponse:
            def __init__(self):
                self.choices = [{"message": {"content": "{}"}}]
                self.usage = {"prompt_tokens": 0, "completion_tokens": 0}

        return MockResponse()