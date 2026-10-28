import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from mvp_runtime import memory_paused


def test_critical_memory_pauses_and_requires_recovery_margin():
    assert memory_paused(475)
    assert memory_paused(750, True)
    assert not memory_paused(1000, True)
    assert not memory_paused(750, False)


def test_latest_result_replaces_old_queue_item():
    from queue import Queue

    from mvp_runtime import latest_put

    q = Queue(maxsize=1)
    q.put("old")
    latest_put(q, "new")
    assert q.get_nowait() == "new" and q.empty()
