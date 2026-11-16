"""AllenPerisomaticCell with the vendored cortical channels (needs a C compiler; skipped otherwise)."""
import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from neurosim import channels, viz
from neurosim import morphology as mo
from neurosim.simulation import AllenPerisomaticCell, CurrentClamp

# A real "Biophysical - perisomatic" parameter set (Allen cell 485909730).
PARAMS = {
    "passive": [{"ra": 100.0, "e_pas": -91.4, "cm": [
        {"section": "soma", "cm": 1.0}, {"section": "axon", "cm": 1.0},
        {"section": "dend", "cm": 2.0}, {"section": "apic", "cm": 2.0}]}],
    "fitting": [{"junction_potential": -14.0, "sweeps": [35]}],
    "conditions": [{"celsius": 34.0, "v_init": -91.4, "erev": [{"section": "soma", "ena": 53.0, "ek": -107.0}]}],
    "genome": [
        {"section": "soma", "name": "gbar_Im", "value": 0.0009016, "mechanism": "Im"},
        {"section": "soma", "name": "gbar_Ih", "value": 8.386e-05, "mechanism": "Ih"},
        {"section": "soma", "name": "gbar_NaTs", "value": 0.417415, "mechanism": "NaTs"},
        {"section": "soma", "name": "gbar_Nap", "value": 0.0008231, "mechanism": "Nap"},
        {"section": "soma", "name": "gbar_K_P", "value": 0.0413635, "mechanism": "K_P"},
        {"section": "soma", "name": "gbar_K_T", "value": 0.0024804, "mechanism": "K_T"},
        {"section": "soma", "name": "gbar_SK", "value": 0.0320782, "mechanism": "SK"},
        {"section": "soma", "name": "gbar_Kv3_1", "value": 0.3435009, "mechanism": "Kv3_1"},
        {"section": "soma", "name": "gbar_Ca_HVA", "value": 0.0008947, "mechanism": "Ca_HVA"},
        {"section": "soma", "name": "gbar_Ca_LVA", "value": 0.0076925, "mechanism": "Ca_LVA"},
        {"section": "soma", "name": "gamma_CaDynamics", "value": 0.000103, "mechanism": "CaDynamics"},
        {"section": "soma", "name": "decay_CaDynamics", "value": 748.17, "mechanism": "CaDynamics"},
        {"section": "soma", "name": "g_pas", "value": 0.000526, "mechanism": ""},
        {"section": "axon", "name": "g_pas", "value": 0.000509, "mechanism": ""},
        {"section": "dend", "name": "g_pas", "value": 2.964e-05, "mechanism": ""},
        {"section": "apic", "name": "g_pas", "value": 3.306e-06, "mechanism": ""},
    ],
}


def test_vendored_mod_files_match_checksums():
    readme = (channels.SOURCE_DIR / "README.md").read_text()
    listed = dict((name, digest) for digest, name in re.findall(r"^([0-9a-f]{64})  (\S+\.mod)$", readme, re.M))
    files = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in channels.SOURCE_DIR.glob("*.mod")}
    assert files == listed and len(files) == 16


@pytest.fixture(scope="module")
def compiled():
    if not channels.available():
        pytest.skip("cortical channels could not be compiled (no C compiler?)")


def test_allen_cell_spikes_realistically(compiled, simple_swc):
    m = mo.load(simple_swc)
    cell = AllenPerisomaticCell(m, PARAMS)
    assert len(cell.sections) == mo.summary(m)["number_of_sections"]  # geometry-aligned
    assert cell.v_offset == 14.0
    rest = cell.run(CurrentClamp(amp=0.0, delay=50, dur=100, tstop=150))
    # model rests near e_pas (-91.4) and is reported in the recording frame (+14 mV)
    assert -82 < rest.v["soma"][-1] < -72
    assert len(rest.spike_times) == 0
    res = cell.run(CurrentClamp(amp=0.4, delay=50, dur=200, tstop=300))
    assert len(res.spike_times) >= 2
    w = viz._first_spike_window(res.t - 50, res.v["soma"], window=(-3, 6))
    t, v = w
    half = v.min() + (v.max() - v.min()) / 2
    width = np.ptp(t[v >= half])
    assert width < 1.5  # cortical spikes are ~1 ms wide; squid HH spikes here are ~4 ms


def test_multi_model_figures():
    t = np.arange(-100, 1200, 0.1)
    v = np.full_like(t, -70.0)
    v[(t > 50) & (t < 51)] = 30.0
    fig = viz.comparison_figure(t, v, [("A", t, v), ("B", t, v - 1)])
    assert [tr.name for tr in fig.data] == ["Recorded (Allen)", "A", "B"]
    assert len(viz.spike_shape_figure(t, v, [("A", t, v)]).data) == 2
    assert len(viz.fi_comparison_figure(np.array([0.1]), np.array([1.0]), {"A": [2.0]}).data) == 2
