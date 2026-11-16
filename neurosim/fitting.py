"""Fit a MorphologyCell to a recorded neuron's electrophysiology.

The model keeps the real morphology and Hodgkin-Huxley channel kinetics and
fits a handful of densities in two cheap, mostly 1-D stages:

1. **Passive** – a uniform leak everywhere, with
   * ``e_pas`` set so the model rests at the measured resting potential,
   * ``g_pas`` found by bisection so the steady-state input resistance
     matches (it does not depend on ``cm``),
   * dendritic ``cm`` then scaled so the membrane time constant matches
     (soma and axon keep 1 uF/cm2; extra dendritic capacitance stands in for
     spines, which reconstructions omit).

2. **Active** – Na and K densities (somatic and AIS densities are scaled
   together) and channel kinetic speed:
   * a small grid over kinetic speed (NEURON's temperature factor) and K
     density, keeping the combination whose f-I curve best matches the
     recorded rates above threshold,
   * bisection on Na density so the model starts firing at the recorded
     rheobase (fires at the rheobase step but not at the step below).

Squid HH channels are not cortical channels, so expect a good match of the
passive properties, rheobase and f-I curve, and only a qualitative match of
spike shape.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from .simulation import Biophysics, CurrentClamp, MorphologyCell

FIT_DT = 0.05  # ms
FIT_D_LAMBDA = 0.3
DELAY = 100.0  # ms before the step in fitting protocols


@dataclass
class Targets:
    vrest: float  # mV
    ri: float  # MOhm
    tau: float  # ms
    fi_amps: np.ndarray  # nA
    fi_rates: np.ndarray  # Hz
    step_ms: float = 1000.0

    @property
    def rheobase(self) -> float | None:
        firing = self.fi_amps[self.fi_rates > 0]
        return float(firing.min()) if len(firing) else None


@dataclass
class FitResult:
    bio: Biophysics
    targets: Targets
    achieved: dict[str, float]
    fi_amps: np.ndarray
    fi_rates_model: np.ndarray
    notes: list[str] = field(default_factory=list)


def targets_from_allen(cell, sweeps) -> Targets:
    """Fitting targets from an :class:`AllenCell` and its sweeps."""
    from .sources.allen import AllenCellTypesClient

    f = cell.features
    amps, rates = AllenCellTypesClient.fi_curve(sweeps)
    durs = [s.duration_s for s in sweeps if s.stimulus == "Long Square" and s.duration_s]
    return Targets(
        vrest=float(f["vrest"]), ri=float(f["ri"]), tau=float(f["tau"]),
        fi_amps=amps, fi_rates=rates, step_ms=1000.0 * (float(np.median(durs)) if durs else 1.0),
    )


# ---------------------------------------------------------------------------
# Measurements on the model
# ---------------------------------------------------------------------------
def passive_response(cell: MorphologyCell, amp: float = -0.03, dur: float = 400.0, dt: float = 0.1):
    """(resting potential mV, input resistance MOhm, time constant ms) from a small hyperpolarising step."""
    res = cell.run(CurrentClamp(amp=amp, delay=DELAY, dur=dur, tstop=DELAY + dur, dt=dt))
    t, v = res.t, res.v["soma"]
    v0 = float(np.interp(DELAY - 1, t, v))
    vss = float(v[-1])
    ri = (vss - v0) / amp  # mV / nA = MOhm
    target = v0 + (1 - 1 / math.e) * (vss - v0)
    after = (t >= DELAY) & ((v - target) * np.sign(amp) >= 0)  # reached 63% of the deflection
    tau = float(t[after][0] - DELAY) if after.any() else float("nan")
    return v0, ri, tau


def firing_rate(cell: MorphologyCell, amp: float, step_ms: float) -> float:
    res = cell.run(CurrentClamp(amp=amp, delay=DELAY, dur=step_ms, tstop=DELAY + step_ms + 20, dt=FIT_DT))
    return res.firing_rate


# ---------------------------------------------------------------------------
# Fitting stages
# ---------------------------------------------------------------------------
def _cell(morph, bio: Biophysics) -> MorphologyCell:
    return MorphologyCell(morph, bio)


def fit_passive(morph, targets: Targets, bio: Biophysics | None = None,
                progress: Callable[[str], None] | None = None) -> tuple[Biophysics, list[str]]:
    notes = []
    say = progress or (lambda msg: None)
    b = replace(bio or Biophysics(), uniform_leak=True, e_pas=targets.vrest, v_init=targets.vrest,
                d_lambda=FIT_D_LAMBDA)

    # 1. g_pas for input resistance (bisection in log space; Ri decreases with g_pas)
    lo, hi = math.log(1e-6), math.log(1e-2)
    ri_at = {}

    def ri_for(lg: float) -> float:
        cell = _cell(morph, replace(b, g_pas=math.exp(lg)))
        ri_at[lg] = passive_response(cell)[1]
        return ri_at[lg]

    if ri_for(lo) < targets.ri:
        notes.append(f"Even a very low leak gives Ri = {ri_at[lo]:.0f} MΩ < {targets.ri:.0f} MΩ measured "
                     "(the reconstruction may be incomplete).")
        lg = lo
    else:
        for i in range(14):
            mid = (lo + hi) / 2
            say(f"Passive fit: input resistance, step {i + 1}/14")
            if ri_for(mid) > targets.ri:
                lo = mid
            else:
                hi = mid
        lg = (lo + hi) / 2
    b = replace(b, g_pas=math.exp(lg))

    # 2. dendritic cm for the time constant (soma/axon keep cm = 1 uF/cm2 so
    #    spikes stay fast), 3. e_pas correction for resting potential
    b = replace(b, cm=1.0, cm_dendrite=b.cm)
    for i in range(4):
        say(f"Passive fit: time constant and resting potential, step {i + 1}/4")
        v0, ri, tau = passive_response(_cell(morph, b))
        cm_d = float(np.clip(b.cm_dendrite * targets.tau / tau, 0.3, 10.0)) if tau > 0 else b.cm_dendrite
        b = replace(b, cm_dendrite=cm_d, e_pas=b.e_pas + (targets.vrest - v0), v_init=targets.vrest)
    if not 0.5 <= b.cm_dendrite <= 3.0:
        notes.append(f"Fitted dendritic capacitance {b.cm_dendrite:.2f} µF/cm² is outside the usual 0.5-3 "
                     "range (spines add up to ~2x); the reconstruction may be missing branches.")
    return b, notes


def _fi_amps_for_fit(targets: Targets) -> tuple[float | None, float | None, list[float]]:
    """(last silent amplitude, rheobase amplitude, supra-threshold amplitudes)."""
    a, r = targets.fi_amps, targets.fi_rates
    rheo = targets.rheobase
    if rheo is None:
        return None, None, []
    below = a[(a < rheo) & (r == 0)]
    silent = float(below.max()) if len(below) else None
    supra = a[a > rheo]
    picks = []
    if len(supra):
        picks = sorted({float(supra.min()), float(supra[len(supra) // 2]), float(supra.max())})
    return silent, rheo, picks


def fit_active(morph, targets: Targets, bio: Biophysics,
               progress: Callable[[str], None] | None = None) -> tuple[Biophysics, list[str]]:
    notes = []
    say = progress or (lambda msg: None)
    silent, rheo, supra = _fi_amps_for_fit(targets)
    if rheo is None:
        notes.append("The recorded cell never fired on long steps; active properties were not fitted.")
        return bio, notes
    measured = dict(zip(targets.fi_amps.round(4), targets.fi_rates))

    def fires(b: Biophysics, amp: float) -> bool:
        return firing_rate(_cell(morph, b), amp, targets.step_ms) > 0

    def fit_na(b: Biophysics, label: str, steps: int = 8) -> Biophysics:
        """Bisection on Na scale: fire at rheobase, stay silent one step below."""
        lo, hi = math.log(0.05), math.log(20.0)
        base = b.gnabar
        for i in range(steps):
            say(f"Active fit: {label}, Na density step {i + 1}/{steps}")
            mid = (lo + hi) / 2
            if fires(replace(b, gnabar=base * math.exp(mid)), rheo):
                hi = mid
            else:
                lo = mid
        return replace(b, gnabar=base * math.exp(hi))

    if not supra:
        b = fit_na(bio, "rheobase")
    else:
        # Squid HH channels jump straight to ~40 Hz at threshold, while cortical
        # neurons can fire slowly. Slowing the channel kinetics (NEURON's
        # temperature factor) and adding K lowers the onset rate, so search a
        # small grid of both, re-fitting Na for the rheobase at each point.
        grid = [(temp, k) for temp in (6.3, 0.0, -6.0) for k in (1.0, 3.0)]
        best, best_err = bio, float("inf")
        for i, (temp, k) in enumerate(grid):
            trial = replace(bio, celsius=temp, gkbar=bio.gkbar * k)
            trial = fit_na(trial, f"kinetics {i + 1}/{len(grid)}", steps=5)
            cell = _cell(morph, trial)
            err = sum((firing_rate(cell, a, targets.step_ms) - measured[round(a, 4)]) ** 2 for a in supra)
            del cell
            if err < best_err:
                best, best_err = trial, err
        b = fit_na(replace(best, gnabar=bio.gnabar), "refine")
    if silent is not None and fires(b, silent):
        notes.append("Model fires slightly below the recorded rheobase.")
    return b, notes


def fit(morph, targets: Targets, bio: Biophysics | None = None,
        progress: Callable[[str], None] | None = None) -> FitResult:
    """Full passive + active fit. Takes ~30-90 s for a typical reconstruction."""
    b, notes = fit_passive(morph, targets, bio, progress)
    b, notes_active = fit_active(morph, targets, b, progress)
    notes += notes_active

    if progress:
        progress("Checking the fitted model")
    cell = _cell(morph, b)
    v0, ri, tau = passive_response(cell)
    rates = np.array([firing_rate(cell, a, targets.step_ms) for a in targets.fi_amps])
    del cell
    firing = targets.fi_amps[rates > 0]
    achieved = {"vrest": v0, "ri": ri, "tau": tau, "rheobase": float(firing.min()) if len(firing) else float("nan")}
    return FitResult(bio=b, targets=targets, achieved=achieved,
                     fi_amps=targets.fi_amps, fi_rates_model=rates, notes=notes)


# ---------------------------------------------------------------------------
# Persistence (fits are slow, so the app stores them on disk)
# ---------------------------------------------------------------------------
def to_dict(res: FitResult) -> dict:
    from dataclasses import asdict

    t = res.targets
    return {
        "bio": asdict(res.bio),
        "targets": {"vrest": t.vrest, "ri": t.ri, "tau": t.tau, "step_ms": t.step_ms,
                    "fi_amps": t.fi_amps.tolist(), "fi_rates": t.fi_rates.tolist()},
        "achieved": res.achieved,
        "fi_rates_model": res.fi_rates_model.tolist(),
        "notes": res.notes,
    }


def from_dict(d: dict) -> FitResult:
    t = d["targets"]
    targets = Targets(vrest=t["vrest"], ri=t["ri"], tau=t["tau"], step_ms=t["step_ms"],
                      fi_amps=np.asarray(t["fi_amps"]), fi_rates=np.asarray(t["fi_rates"]))
    return FitResult(bio=Biophysics(**d["bio"]), targets=targets, achieved=d["achieved"],
                     fi_amps=targets.fi_amps, fi_rates_model=np.asarray(d["fi_rates_model"]),
                     notes=list(d["notes"]))
