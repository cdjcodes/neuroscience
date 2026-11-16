import numpy as np
import pytest

from neurosim import morphology as mo
from neurosim import viz
from neurosim.simulation import CurrentClamp, MorphologyCell


@pytest.fixture
def geom_and_result(simple_swc):
    m = mo.load(simple_swc)
    res = MorphologyCell(m).run(CurrentClamp(amp=0.3, tstop=60), record_sections_dt=1.0)
    return mo.geometry(m), res


def test_morphology_figure(geom_and_result):
    geom, _ = geom_and_result
    fig = viz.morphology_figure(geom)
    assert {t.name for t in fig.data} == {"Axon", "Basal dendrite", "Apical dendrite", "Soma"}


def test_voltage_figure_frames(geom_and_result):
    geom, res = geom_and_result
    fig = viz.voltage_figure(geom, res, n_frames=10)
    assert len(fig.frames) == 10
    colors = np.asarray(fig.frames[-1].data[0].line.color, dtype=float)
    assert np.nanmin(colors) < -40 or np.nanmax(colors) > -80  # real voltages, not placeholders


def test_voltage_figure_requires_section_recording(simple_swc, geom_and_result):
    geom, _ = geom_and_result
    res = MorphologyCell(mo.load(simple_swc)).run(CurrentClamp(tstop=20))
    with pytest.raises(ValueError):
        viz.voltage_figure(geom, res)


def test_trace_and_summary_figures(geom_and_result):
    _, res = geom_and_result
    assert len(viz.traces_figure(res).data) >= len(res.v) + 1
    viz.fi_figure(np.array([0, 1.0]), np.array([0, 10.0]))
    viz.sholl_figure(np.arange(3.0), np.array([1, 2, 1]))
    viz.histogram_figure(np.array([1.0, np.nan, 3.0]), "x")
