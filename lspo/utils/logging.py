from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional
class RunLogger:
    def __init__(self, output_dir: str | Path, run_name: str, verbose: bool = True):
        self.dir = Path(output_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{run_name}.jsonl"
        self.verbose = verbose
        self._start = time.time()
        self._handle = self.path.open("a", encoding="utf-8")
    def log(self, record: Dict[str, Any], step: Optional[int] = None) -> None:
        payload = dict(record)
        if step is not None:
            payload["step"] = step
        payload["wall_clock_s"] = round(time.time() - self._start, 3)
        self._handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._handle.flush()
        if self.verbose:
            head = f"[step {step:>5}] " if step is not None else ""
            body = "  ".join(
                f"{k}={_fmt(v)}" for k, v in record.items() if not k.startswith("_")
            )
            print(head + body, flush=True)
    def info(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)
    def close(self) -> None:
        try:
            self._handle.close()
        except Exception:
            pass
    def __enter__(self) -> "RunLogger":
        return self
    def __exit__(self, *exc) -> None:
        self.close()
def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
def load_records(path: str | Path) -> list:
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records
