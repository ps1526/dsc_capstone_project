"""
Inference utilities for AR-Bench

This module provides inference capabilities for different LLM providers including OpenAI and Together AI.
"""
import os
from typing import Dict, List, Optional

import numpy as np
from openai import OpenAI


_default_model = "gpt-4o"
_default_temperature = 0.7
_default_top_p = 0.7

def inference(messages: List[Dict], model: Optional[str] = None, json_format: bool = False, api_key: Optional[str] = None, base_url: Optional[str] = None,
              temperature: Optional[float] = None, top_p: Optional[float] = None) -> str:

    try:
        # Use global defaults if parameters are not specified
        if model is None:
            model = _default_model
        if temperature is None:
            temperature = _default_temperature
        if top_p is None:
            top_p = _default_top_p
            
        client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p
        }
        
        if json_format:
            kwargs["response_format"] = {"type": "json_object"}
        
        response = client.chat.completions.create(**kwargs)
        return response or "{}"
    except Exception as e:
        print(f"Inference error: {e}")
        return "{}"

