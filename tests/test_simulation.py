import numpy as np
import pytest

from neurosim import morphology as mo
from neurosim.simulation import (
    BallAndStick, Biophysics, CurrentClamp, MorphologyCell, detect_spikes, fi_curve, rheobase,
)


def test_detect_spikes():
    t = np.arange(10.0)
    v = np.array([-65, -20, 10, 20, -60, -65, 5, -60, -65, -65.0])
    assert detect_spikes(t, v).tolist() == [2.0, 6.0]


def test_ball_and_stick_rests_and_fires():
    cell = BallAndStick()
    rest = cell.run(CurrentClamp(amp=0.0, tstop=50))
    assert rest.v["soma"][-1] == pytest.approx(-65, abs=1)
    assert len(rest.spike_times) == 0

    res = cell.run(CurrentClamp(amp=0.2))
    assert len(res.spike_times) > 1
    # spikes only during the stimulus
    assert res.spike_times.min() >= res.stim.delay
    # passive dendrite attenuates the spike
    assert res.v["dendrite_tip"].max() < res.v["soma"].max()


def test_fi_curve_is_monotonic_at_low_currents():
    amps, rates = fi_curve(BallAndStick(), [0.0, 0.05, 0.1, 0.2])
    assert rates[0] == 0 and np.all(np.diff(rates) >= 0) and rates[-1] > 0


def test_rheobase_ball_and_stick():
    rb = rheobase(BallAndStick(), tol=0.005)
    assert 0 < rb < 0.2


def test_morphology_cell_matches_geometry(simple_swc):
    m = mo.load(simple_swc)
    cell = MorphologyCell(m)
    assert len(cell.sections) == mo.summary(m)["number_of_sections"]
    assert {"soma", "farthest_dendrite", "axon"} <= set(cell.sites)
    res = cell.run(CurrentClamp(amp=0.3), record_sections_dt=1.0)
    assert len(res.spike_times) >= 1
    assert res.v_sections.shape == (len(cell.sections), len(res.t_sections))


def test_morphology_cell_without_axon_gets_synthetic_ais(branched_soma_swc):
    cell = MorphologyCell(mo.load(branched_soma_swc))
    assert cell.ais is not None
    assert cell.run(CurrentClamp(amp=0.5)).spike_times.size >= 1


def test_passive_cell_does_not_fire(simple_swc):
    bio = Biophysics(gnabar=0.0)
    res = MorphologyCell(mo.load(simple_swc), bio).run(CurrentClamp(amp=0.5))
    assert res.spike_times.size == 0
