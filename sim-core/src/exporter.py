from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable


def ensureParent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def writeScenario(payload: Dict[str, object], outputPath: str) -> None:
    ensureParent(outputPath)
    with Path(outputPath).open("w", encoding="utf-8") as fileObj:
        json.dump(payload, fileObj, ensure_ascii=False, indent=2)


def writeJson(data: Dict[str, object], outputPath: str) -> None:
    ensureParent(outputPath)
    with Path(outputPath).open("w", encoding="utf-8") as fileObj:
        json.dump(data, fileObj, ensure_ascii=False, indent=2)
