"""Format-agnostic landing-zone loader.

IDAMP only ever read CSV. RADAR's landing zone accepts three formats — CSV,
JSON, and Excel (XLSX) — because real returns pipelines rarely arrive as a
single tidy format: order exports come as CSV, a returns-processing system
dumps JSON, and the product master is a hand-maintained spreadsheet. Every
agent downstream of this loader works on a plain DataFrame, so nothing else
in the pipeline needs to know which format a file started as.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from core.config import SUPPORTED_LANDING_FORMATS


def load_any(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_LANDING_FORMATS:
        raise ValueError(
            f"Unsupported file format '{suffix}' for {path.name}. "
            f"Supported formats: {SUPPORTED_LANDING_FORMATS}"
        )

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Accept either a top-level list of records, or {"data": [...]} / {"records": [...]}
        if isinstance(raw, dict):
            for key in ("data", "records", "items"):
                if key in raw and isinstance(raw[key], list):
                    raw = raw[key]
                    break
            else:
                raw = [raw]
        return pd.json_normalize(raw)

    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)

    raise ValueError(f"Unhandled format '{suffix}'")


def describe_format(path: str | Path) -> str:
    return Path(path).suffix.lower().lstrip(".")
