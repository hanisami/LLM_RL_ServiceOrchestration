from __future__ import annotations

import csv
import os
from typing import Dict, List


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    ensure_dir(os.path.dirname(path))
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
