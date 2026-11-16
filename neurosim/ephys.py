"""Electrophysiology features with eFEL (https://github.com/openbraininstitute/eFEL).

The same function is applied to recorded (Allen) and simulated traces so
the two can be compared like-for-like.
"""

from __future__ import annotations

import math

import numpy as np

# eFEL feature -> (label, unit); per-spike features are averaged
FEATURES: dict[str, tuple[str, str]] = {
    "voltage_base": ("Resting potential", "mV"),
    "ohmic_input_resistance_vb_ssse": ("Input resistance", "MΩ"),
    "Spikecount": ("Spike count", ""),
    "mean_frequency": ("Mean firing rate", "Hz"),
    "time_to_first_spike": ("First-spike latency", "ms"),
    "AP_begin_voltage": ("Spike threshold", "mV"),
    "AP_height": ("Spike peak", "mV"),
    "AP_amplitude": ("Spike amplitude", "mV"),
    "spike_half_width": ("Spike half-width", "ms"),
    "AHP_depth_abs": ("After-hyperpolarisation", "mV"),
    "adaptation_index2": ("Adaptation index", ""),
    "ISI_CV": ("ISI coefficient of variation", ""),
    "steady_state_voltage_stimend": ("Voltage at end of step", "mV"),
}


def features(
    t: np.ndarray,
    v: np.ndarray,
    stim_start: float,
    stim_end: float,
    amp_na: float | None = None,
    names: list[str] | None = None,
) -> dict[str, float]:
    """eFEL features of one current-step trace (time in ms, voltage in mV).

    Per-spike features are averaged. Features eFEL cannot compute (e.g. spike
    shape when there are no spikes) are returned as NaN.
    """
    import efel

    names = list(names or FEATURES)
    t = np.asarray(t, dtype=float)
    offset = -t[0]  # eFEL measures the baseline from t=0 up to stim_start
    trace = {
        "T": t + offset,
        "V": np.asarray(v, dtype=float),
        "stim_start": [stim_start + offset],
        "stim_end": [stim_end + offset],
    }
    if amp_na is not None:
        trace["stimulus_current"] = [amp_na]
    else:
        names = [n for n in names if n != "ohmic_input_resistance_vb_ssse"]
    raw = efel.get_feature_values([trace], names, raise_warnings=False)[0]
    out = {}
    for n in names:
        val = raw.get(n)
        out[n] = float(np.mean(val)) if val is not None and len(val) else float("nan")
    if "Spikecount" in out and math.isnan(out["Spikecount"]):
        out["Spikecount"] = 0.0
    return out


def compare(measured: dict[str, float], simulated: dict[str, float]) -> list[dict]:
    """Rows for a measured-vs-simulated feature table."""
    rows = []
    for name, (label, unit) in FEATURES.items():
        if name not in measured and name not in simulated:
            continue
        m, s = measured.get(name, float("nan")), simulated.get(name, float("nan"))
        rows.append({
            "Feature": label,
            "Unit": unit,
            "Recorded": m,
            "Simulated": s,
            "Difference": s - m if not (math.isnan(m) or math.isnan(s)) else float("nan"),
        })
    return rows
