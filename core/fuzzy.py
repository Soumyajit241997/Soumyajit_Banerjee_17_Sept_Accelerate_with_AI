"""Fuzzy name matching used to tolerate LLM-authored table names that don't
exactly match the deterministic name RADAR expects (e.g. STTM target_table
"order_transactions" vs the uploaded file's stem "orders"). Exact match is
always tried first; this is only a fallback."""
from __future__ import annotations

import difflib


def best_match(name: str, candidates: list[str], cutoff: float = 0.5) -> str | None:
    """Return the candidate closest to `name`, or None if nothing clears `cutoff`.
    Tries an exact (case-insensitive) match first, then substring containment
    either direction, then difflib's sequence-similarity ratio."""
    if not name or not candidates:
        return None

    low = name.strip().lower()
    lower_map = {c.lower(): c for c in candidates}

    if low in lower_map:
        return lower_map[low]

    for cand_low, cand in lower_map.items():
        if cand_low in low or low in cand_low:
            return cand

    matches = difflib.get_close_matches(low, list(lower_map.keys()), n=1, cutoff=cutoff)
    return lower_map[matches[0]] if matches else None
