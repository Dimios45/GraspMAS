"""In-process message bus.

Single-process by design: the whole pipeline shares one local VLM, so agent
turns are serialized anyway (see docs/GMAS_RUNTIME.md). Handlers may be sync
or async; publish() awaits async handlers in registration order.
"""

import asyncio
import fnmatch
from typing import Awaitable, Callable, Dict, List, Union

from .messages import Msg

Handler = Callable[[Msg], Union[None, Awaitable[None]]]


class MessageBus:
    def __init__(self):
        self._subs: List[tuple[str, Handler]] = []

    def subscribe(self, pattern: str, handler: Handler) -> None:
        """Subscribe to topics matching a glob pattern ('*' for everything)."""
        self._subs.append((pattern, handler))

    async def publish(self, msg: Msg) -> None:
        for pattern, handler in self._subs:
            if fnmatch.fnmatch(msg.topic, pattern):
                out = handler(msg)
                if asyncio.iscoroutine(out):
                    await out
