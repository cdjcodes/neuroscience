import math

import numpy as np
import pytest

from neurosim import ephys, fitting
from neurosim import morphology as mo
from neurosim.simulation import Biophysics, CurrentClamp, MorphologyCell


def test_features_on_simulated_trace(simple_swc):
    res = MorphologyCell(mo.load(simple_swc)).run(CurrentClamp(amp=0.3, delay=50, dur=200, tstop=300))
    f = ephys.features(res.t, res.v["soma"], 50, 250)
    assert f["Spikecount"] == len(res.spike_times) > 0
    assert f["voltage_base"] == pytest.approx(res.v["soma"][(res.t > 40) & (res.t < 50)].mean(), abs=0.5)
    assert f["mean_frequency"] > 0


def test_features_without_spikes_are_nan():
    t = np.arange(0, 300, 0.1)
    v = np.where((t > 50) & (t < 250), -70.0, -75.0)
    f = ephys.features(t, v, 50, 250, amp_na=0.05)
    assert f["Spikecount"] == 0
    assert math.isnan(f["AP_height"])
    assert f["ohmic_input_resistance_vb_ssse"] == pytest.approx(100, rel=0.05)  # 5 mV / 0.05 nA


def test_compare_rows():
    rows = ephys.compare({"Spikecount": 3.0}, {"Spikecount": 5.0})
    assert rows == [{"Feature": "Spike count", "Unit": "", "Recorded": 3.0, "Simulated": 5.0, "Difference": 2.0}]


def test_passive_fit_recovers_known_cell(simple_swc):
    """Generate targets from a model with known passive parameters and fit them back."""
    m = mo.load(simple_swc)
    truth = Biophysics(uniform_leak=True, g_pas=1e-4, e_pas=-75.0, v_init=-75.0, cm_dendrite=2.0,
                       d_lambda=fitting.FIT_D_LAMBDA)
    v0, ri, tau = fitting.passive_response(MorphologyCell(m, truth))
    targets = fitting.Targets(vrest=v0, ri=ri, tau=tau, fi_amps=np.array([0.1]), fi_rates=np.array([0.0]))
    b, _ = fitting.fit_passive(m, targets)
    v0f, rif, tauf = fitting.passive_response(MorphologyCell(m, b))
    assert v0f == pytest.approx(v0, abs=0.5)
    assert rif == pytest.approx(ri, rel=0.03)
    assert tauf == pytest.approx(tau, rel=0.08)


def test_full_fit_matches_rheobase_and_roundtrips(simple_swc):
    m = mo.load(simple_swc)
    targets = fitting.Targets(
        vrest=-72.0, ri=150.0, tau=15.0,
        fi_amps=np.array([0.02, 0.04, 0.06, 0.08, 0.10]),
        fi_rates=np.array([0.0, 0.0, 5.0, 10.0, 15.0]),
    )
    res = fitting.fit(m, targets)
    assert res.achieved["rheobase"] == pytest.approx(0.06)
    assert res.fi_rates_model[:2].tolist() == [0.0, 0.0]
    back = fitting.from_dict(fitting.to_dict(res))
    assert back.bio == res.bio and np.allclose(back.fi_rates_model, res.fi_rates_model)
