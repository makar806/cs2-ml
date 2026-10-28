import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark_tracking import run


@pytest.mark.parametrize("hz", [30, 45, 60])
@pytest.mark.parametrize("speed", [0, 180, -180])
def test_delayed_multirate_loop_tracks_without_divergence_and_smaller_steps(hz, speed):
    instant = run(hz, speed, False)
    smooth = run(hz, speed, True)
    assert smooth["max_error_px"] < 18
    assert smooth["mean_error_px"] < instant["mean_error_px"] + 3

    assert 1 <= smooth["shots"] <= 8
    if speed:
        assert smooth["largest_input_counts"] < instant["largest_input_counts"]
        assert smooth["input_events"] > instant["input_events"]
