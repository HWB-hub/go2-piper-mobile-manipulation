"""Measure actual forward motion and support through a walk/run cycle."""
import json
import sys
from pathlib import Path

import numpy as np


def analyze(folder):
    data = np.load(folder / "gait_samples.npz")
    motion = np.load(folder / "samples.npz")
    gait, values = data["values"], motion["values"]
    dt = float(data["dt"])
    time = float(data["start_time"]) + dt * np.arange(len(gait))
    vx = values[..., list(motion["columns"]).index("vx")]
    instant = "instant_contact_FL" in data["columns"]
    contact = gait[..., 25:29] if instant else gait[..., 4:8]
    result = {}
    for label, (start, end) in {
        "walk": (4.5, 7.), "accelerate": (7., 12.), "run": (13., 22.),
        "decelerate": (22., 27.), "walk_again": (28., 37.),
    }.items():
        window = (time >= start) & (time < end)
        velocity = vx[window]
        valid = np.isfinite(velocity)
        feet = contact[window]
        flight = valid & (feet.sum(axis=-1) == 0) & (gait[window, :, :4].min(axis=-1) > .006)
        # Require two consecutive samples to reject isolated contact gaps.
        verified_flight = np.zeros_like(flight)
        verified_flight[1:] = flight[1:] & flight[:-1]
        verified_flight[:-1] |= flight[1:] & flight[:-1]
        result[label] = {
            "forward_speed_mean_m_s": float(velocity[valid].mean()) if valid.any() else None,
            "forward_speed_p95_m_s": float(np.quantile(velocity[valid], .95)) if valid.any() else None,
            "support_fraction": feet[valid].mean(axis=0).tolist() if valid.any() else None,
            "flight_fraction": float(verified_flight.sum() / valid.sum()) if valid.any() else None,
            "episodes_with_flight": int(verified_flight.any(axis=0).sum()),
            "episodes_with_single_sample_flight": int(flight.any(axis=0).sum()),
            "uses_instant_contact": instant,
            "reference_frequency_hz": float(gait[window, :, 23][valid].mean()) if valid.any() else None,
            "valid_samples": int(valid.sum()),
        }
    (folder / "walk_run_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    for name in sys.argv[1:]:
        print(name, json.dumps(analyze(Path(__file__).parent / "runs" / name)))
