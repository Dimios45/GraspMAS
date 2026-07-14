"""Structured audit log: one JSONL file per session.

Subscribes to '*' on the bus so every typed message is recorded verbatim.
The eval tables in gmas/eval/report.py are built from these files.
"""

import json
import os
from typing import Optional

from .messages import Msg


class AuditLog:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._f = open(path, "a")
        self.path = path

    def __call__(self, msg: Msg) -> None:
        self._f.write(msg.model_dump_json() + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()
