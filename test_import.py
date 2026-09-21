"""Quick import validation: fails fast if a module is broken or a dependency
is missing, without needing an LLM API key."""
import core.audit
import core.charts
import core.config
import core.forecasting
import core.json_utils
import core.loaders
import core.memory
import core.observability
import core.preview
import core.state
import core.transforms

import agents.bronze_agent
import agents.enrichment_agent
import agents.gold_agent
import agents.orchestrator
import agents.profiler
import agents.reporter
import agents.silver_agent
import agents.sttm_generator

print("All RADAR modules imported successfully.")
