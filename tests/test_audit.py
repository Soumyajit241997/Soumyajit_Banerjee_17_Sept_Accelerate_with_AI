from core.audit import get_logs, log_event


def test_log_event_roundtrip(tmp_path, monkeypatch):
    import core.audit as audit_module

    monkeypatch.setattr(audit_module, "AUDIT_DIR", tmp_path)
    log_event("run_x", "profiler", "profile_generated", profile_path="foo.json")
    logs = get_logs("run_x")
    assert len(logs) == 1
    assert logs[0]["agent"] == "profiler"
    assert logs[0]["action"] == "profile_generated"
    assert logs[0]["profile_path"] == "foo.json"


def test_get_logs_missing_run_returns_empty():
    assert get_logs("does_not_exist_run") == []
