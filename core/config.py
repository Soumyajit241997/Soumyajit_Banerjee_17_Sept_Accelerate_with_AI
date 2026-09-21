"""Central configuration for RADAR.

RADAR = Returns Analytics via Data-driven Agentic Reasoning.

Mirrors IDAMP's config pattern: paths, LLM provider selection, directory
bootstrap. The one structural addition is SIGNAL_DIR, the storage location
for the new Signal (feature-engineering) layer that sits between Silver and
Gold in this pipeline.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
LANDING_DIR = DATA_DIR / "landing"
PROFILE_DIR = DATA_DIR / "profiles"
STTM_DIR = DATA_DIR / "sttm"
BRONZE_DIR = DATA_DIR / "bronze_layer"
SILVER_DIR = DATA_DIR / "silver_layer"
SIGNAL_DIR = DATA_DIR / "signal_layer"  # NEW: feature-engineering layer (Silver -> Signal -> Gold)
GOLD_DIR = DATA_DIR / "gold_layer"
TRACE_DIR = DATA_DIR / "traces"

REPORTS_DIR = BASE_DIR / "reports"
AUDIT_DIR = BASE_DIR / "audit_logs"
CHROMA_DIR = BASE_DIR / "chroma_store"
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"

ALL_DIRS = [
    LANDING_DIR, PROFILE_DIR, STTM_DIR, BRONZE_DIR, SILVER_DIR,
    SIGNAL_DIR, GOLD_DIR, TRACE_DIR, REPORTS_DIR, AUDIT_DIR, CHROMA_DIR,
]

for _d in ALL_DIRS:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Supported landing file formats
# ---------------------------------------------------------------------------
# IDAMP only ever ingested CSV. RADAR additionally accepts JSON (e.g. a
# returns-processing export) and Excel/XLSX (e.g. a product master maintained
# by hand in a spreadsheet). Each loader normalises to a pandas DataFrame so
# every downstream agent stays format-agnostic.
SUPPORTED_LANDING_FORMATS = (".csv", ".json", ".xlsx", ".xls")

# ---------------------------------------------------------------------------
# LLM configuration (Anthropic Claude preferred, then Groq, then Gemini)
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def get_llm(temperature: float = 0.1):
    """Return a chat model instance: Anthropic Claude if ANTHROPIC_API_KEY is set,
    else Groq, else Gemini."""
    if ANTHROPIC_API_KEY:
        from langchain_anthropic import ChatAnthropic

        # Newer Claude models (e.g. claude-sonnet-5) reject `temperature` outright
        # ("temperature is deprecated for this model") rather than ignoring it.
        return ChatAnthropic(model=ANTHROPIC_MODEL, api_key=ANTHROPIC_API_KEY)
    if GROQ_API_KEY:
        from langchain_groq import ChatGroq

        return ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=temperature)
    if GOOGLE_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GOOGLE_API_KEY, temperature=temperature)
    raise RuntimeError(
        "No LLM configured. Set ANTHROPIC_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file."
    )


# ---------------------------------------------------------------------------
# Forecasting configuration (Reporter agent)
# ---------------------------------------------------------------------------
FORECAST_HORIZON_MONTHS = int(os.getenv("FORECAST_HORIZON_MONTHS", "3"))
FORECAST_MIN_HISTORY_MONTHS = 3  # minimum months of history needed to trust a trend fit
