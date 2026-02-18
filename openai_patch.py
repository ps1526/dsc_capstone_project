"""Patch OpenAI client to use longer timeouts"""
import sys
from unittest.mock import patch

# Patch before any imports
original_import = __builtins__.__import__

def patched_import(name, *args, **kwargs):
    module = original_import(name, *args, **kwargs)
    
    if name == 'openai' or (hasattr(module, '__name__') and module.__name__ == 'openai'):
        try:
            from openai import OpenAI
            original_init = OpenAI.__init__
            
            def new_init(self, *args, **kwargs):
                kwargs['timeout'] = 300.0  # 5 minutes
                kwargs['max_retries'] = 5
                return original_init(self, *args, **kwargs)
            
            OpenAI.__init__ = new_init
            print("✅ Patched OpenAI client: timeout=300s, max_retries=5", file=sys.stderr)
        except:
            pass
    
    return module

__builtins__.__import__ = patched_import
