"""Wrapper to patch OpenAI client timeouts"""
import openai
from openai import OpenAI as _OriginalOpenAI

class OpenAI(_OriginalOpenAI):
    def __init__(self, *args, **kwargs):
        # Force longer timeout and retries
        kwargs['timeout'] = kwargs.get('timeout', 300.0)
        kwargs['max_retries'] = kwargs.get('max_retries', 5)
        super().__init__(*args, **kwargs)

# Replace the OpenAI class
openai.OpenAI = OpenAI

print("✅ OpenAI client patched with timeout=300s, max_retries=5")
