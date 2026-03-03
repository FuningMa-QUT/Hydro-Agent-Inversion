import os

V3 = "deepseek-chat"
R1 = "deepseek-reasoner"

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
    "cache_seed": None
}
