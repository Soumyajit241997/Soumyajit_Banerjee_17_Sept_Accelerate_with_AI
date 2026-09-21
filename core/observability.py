"""Agent observability: captures a structured trace for every agent invocation."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from core.config import TRACE_DIR


class AgentTrace:
    def __init__(self, agent: str, run_id: str):
        self.agent = agent
        self.run_id = run_id
        self.start_time = time.time()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.steps: list[dict] = []
        self.status = "running"
        self.error: str | None = None

    def record_step(self, kind: str, content) -> None:
        self.steps.append({"kind": kind, "content": _safe(content)})

    def finish(self, status: str = "success", error: str | None = None) -> dict:
        self.status = status
        self.error = error
        duration = round(time.time() - self.start_time, 3)
        trace = {
            "agent": self.agent,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "duration_seconds": duration,
            "status": self.status,
            "error": self.error,
            "steps": self.steps,
        }
        out_path = TRACE_DIR / f"{self.run_id}_{self.agent}_{int(self.start_time)}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2, default=str)
        print(f"[TRACE] {self.agent} ({self.run_id}) -> {self.status} in {duration}s")
        return trace

    def from_messages(self, messages: list) -> None:
        """Extract plan/tool-call/observation steps from a LangGraph message history."""
        for msg in messages:
            cls_name = msg.__class__.__name__
            if cls_name == "HumanMessage":
                self.record_step("goal", getattr(msg, "content", ""))
            elif cls_name == "AIMessage":
                content = getattr(msg, "content", "")
                if content:
                    self.record_step("reasoning", content)
                for call in getattr(msg, "tool_calls", []) or []:
                    self.record_step("tool_call", {"name": call.get("name"), "args": call.get("args")})
            elif cls_name == "ToolMessage":
                self.record_step("tool_result", getattr(msg, "content", ""))


def _safe(content):
    try:
        json.dumps(content, default=str)
        return content
    except Exception:
        return str(content)
