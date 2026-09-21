"""Small, testable transformation primitives shared by the Bronze, Silver, and
Signal agents. Kept dependency-free of the agent/LLM layer so they can be
unit-tested directly."""
from __future__ import annotations

import pandas as pd


def cast_series(series: pd.Series, type_hint: str) -> pd.Series:
    hint = (type_hint or "").lower()
    if "int" in hint:
        return pd.to_numeric(series, errors="coerce").astype("Int64")
    if "float" in hint or "double" in hint or "decimal" in hint or "numeric" in hint:
        return pd.to_numeric(series, errors="coerce").astype(float)
    if "date" in hint or "time" in hint:
        return pd.to_datetime(series, errors="coerce")
    return series.astype("string")


def standardize_date(series: pd.Series) -> pd.Series:
    # format="mixed" lets pandas infer the format per-element, so a column with
    # inconsistent date formats (e.g. 01/15/2025 alongside 2025-02-20) still parses.
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    return parsed.dt.strftime("%Y-%m-%d")


def normalize_text(series: pd.Series, logic_hint: str) -> pd.Series:
    hint = (logic_hint or "").lower()
    text = series.astype("string")
    if "lower" in hint:
        text = text.str.lower()
    elif "upper" in hint:
        text = text.str.upper()
    elif "title" in hint:
        text = text.str.title()
    if "strip" in hint or "trim" in hint:
        text = text.str.strip()
    return text


def fill_nulls(series: pd.Series, logic_hint: str) -> pd.Series:
    hint = (logic_hint or "").lower().strip()
    if hint.startswith("fill:"):
        strategy = hint.split(":", 1)[1].strip()
    else:
        strategy = hint

    if strategy in ("mean",) and pd.api.types.is_numeric_dtype(series):
        return series.fillna(series.mean())
    if strategy in ("median",) and pd.api.types.is_numeric_dtype(series):
        return series.fillna(series.median())
    if strategy in ("mode",):
        mode = series.mode(dropna=True)
        return series.fillna(mode.iloc[0] if not mode.empty else "unknown")
    if strategy in ("0", "zero") and pd.api.types.is_numeric_dtype(series):
        return series.fillna(0)
    return series.fillna("unknown")


def add_surrogate_key(df: pd.DataFrame, key_name: str) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    df.insert(0, key_name, range(1, len(df) + 1))
    return df


def preserve_missing_id_columns(source_df: pd.DataFrame, out_df: pd.DataFrame) -> pd.DataFrame:
    """Structural safety net for Bronze/Silver execution: no matter what STTM
    transformation_type an LLM used (or a bug in how we interpret one), a
    business/join-key column present in the source must never silently vanish
    from the output — every later join depends on it. Returns the list of
    column names that had to be recovered this way, for audit logging.

    We've hit this from two different root causes so far: the LLM omitting the
    row entirely, and the LLM using transformation_type="dedup" on the id
    column itself, which our code treated as pure metadata and never copied
    into the output. This closes the whole class rather than one instance.
    """
    recovered = []
    for col in source_df.columns:
        if col.endswith("_id") and not col.startswith("pk_") and col not in out_df.columns:
            out_df[col] = source_df[col]
            recovered.append(col)
    return out_df, recovered
