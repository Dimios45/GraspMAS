import asyncio
from typing import Any
from local_vlm import vlm_chat


class _Choice:
    def __init__(self, text):
        self.message = type("M", (), {"content": text})()


class _Response:
    def __init__(self, text):
        self.choices = [_Choice(text)]


class _Completions:
    async def create(self, model, messages, max_tokens=300, **kwargs):
        text = await asyncio.to_thread(vlm_chat, messages, max_tokens)
        return _Response(text)


class _Chat:
    def __init__(self):
        self.completions = _Completions()


class _LocalClient:
    def __init__(self):
        self.chat = _Chat()
        self.api_key = "EMPTY"


class OpenAILLM:
    def __init__(self, api_file):
        with open(api_file) as f:
            f.readline()  # read but ignore — key not needed for local model
        self.client = _LocalClient()
        self.system_prompt = "Please answer follows retrictly the provide format."

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass
