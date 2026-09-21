"""Parquet schema/sample preview, used to build the context string handed to
each STTM generation stage (Silver, Signal, Gold) about the layer beneath it."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def preview_parquet_files(paths: list[str]) -> str:
    preview = {}
    for p in paths:
        df = pd.read_parquet(p)
        preview[Path(p).name] = {
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "row_count": len(df),
            "sample": json.loads(df.head(3).to_json(orient="records", date_format="iso")),
        }
    return json.dumps(preview, indent=2, default=str)
