# -*- coding: utf-8 -*-
import os

"""
Configuration for LLM agents.
Ensure the OPENAI_API_KEY is set in your system environment variables.
"""

MY_API_KEY = os.getenv("OPENAI_API_KEY")
MY_BASE_URL = "https://api.vectorengine.ai/v1"

if MY_API_KEY is None:
    raise EnvironmentError("OPENAI_API_KEY not found in environment variables.")

config_list = [
    {
        "model": "deepseek-chat", 
        "base_url": MY_BASE_URL,
        "api_key": MY_API_KEY,
        "api_type": "openai",
        "stream": False, 
    }
]

llm_config = {
    "seed": 42,
    "temperature": 0,
    "config_list": config_list,
    "timeout": 600,
}
