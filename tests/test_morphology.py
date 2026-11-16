import numpy as np
import pytest

from neurosim import morphology as mo


def test_load_from_text_and_summary(simple_swc):
    m = mo.load(simple_swc)
    s = mo.summary(m)
    assert s["number_of_neurites"] == 3
    assert s["number_of_bifurcations"] == 1
    assert s["total_length"] == pytest.approx(52 + 2 * np.hypot(40, 40) + 280 + 390, rel=1e-3)
    assert set(mo.per_neurite_type(m)) == {"axon", "basal_dendrite", "apical_dendrite"}


def test_load_from_path(tmp_path, simple_swc):
    p = tmp_path / "cell.swc"
    p.write_text(simple_swc)
    assert mo.summary(mo.load(p))["number_of_neurites"] == 3


def test_branched_soma_is_repaired(branched_soma_swc):
    with pytest.raises(Exception):
        mo.load(branched_soma_swc, repair=False)
    m = mo.load(branched_soma_swc)
    assert mo.summary(m)["number_of_neurites"] == 2


def test_missing_soma_is_created():
    swc = "1 3 0 0 0 1 -1\n2 3 50 0 0 1 1\n3 3 90 30 0 1 2\n4 3 90 -30 0 1 2\n"
    m = mo.load(swc)
    assert len(m.soma.points) == 1
    assert mo.summary(m)["number_of_bifurcations"] == 1


def test_sholl_and_distributions(simple_swc):
    m = mo.load(simple_swc)
    radii, counts = mo.sholl(m, step_size=50)
    assert len(radii) == len(counts) and counts.max() >= 2
    d = mo.distributions(m)
    assert len(d["section_lengths"]) == mo.summary(m)["number_of_sections"]


def test_geometry_has_section_index(simple_swc):
    m = mo.load(simple_swc)
    g = mo.geometry(m)
    assert set(g.lines) == {"axon", "basal_dendrite", "apical_dendrite"}
    for name, pts in g.lines.items():
        assert pts.shape[0] == len(g.widths[name]) == len(g.section_index[name])
    used = np.concatenate([i[i >= 0] for i in g.section_index.values()])
    assert sorted(set(used)) == list(range(int(mo.summary(m)["number_of_sections"])))
