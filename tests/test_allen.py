"""Live tests against the Allen Cell Types API. Skip with ``pytest -m 'not network'``."""
import numpy as np
import pytest

from neurosim import morphology as mo
from neurosim.sources import AllenCellTypesClient

pytestmark = pytest.mark.network
SPECIMEN = 502359001  # mouse VISp L2/3 spiny cell with a full reconstruction


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    return AllenCellTypesClient(cache_dir=tmp_path_factory.mktemp("allen"))


def test_cell_catalogue(client):
    cells = client.cells(species="mouse", reconstructed=True)
    assert len(cells) > 100
    assert all(c.swc_file_id and c.nwb_file_id for c in cells)
    cell = client.get(SPECIMEN)
    assert cell.dendrite_type == "spiny" and "vrest" in cell.features


def test_sweeps_and_fi(client):
    sweeps = client.sweeps(SPECIMEN)
    assert any(s.stimulus == "Long Square" for s in sweeps)
    amps, rates = client.fi_curve(sweeps)
    assert np.all(np.diff(amps) > 0) and rates[0] == 0 and rates.max() > 10


def test_reconstruction_loads(client):
    m = mo.load(client.fetch_swc(SPECIMEN))
    assert "apical_dendrite" in mo.per_neurite_type(m)


def test_stream_one_sweep(client):
    sweep = next(s for s in client.sweeps(SPECIMEN) if s.stimulus == "Long Square" and s.num_spikes >= 5)
    tr = client.trace(SPECIMEN, sweep)
    assert tr.t[0] == pytest.approx(-100, abs=0.1) and tr.t[-1] == pytest.approx(1200, abs=1)
    above = tr.v >= 0
    crossings = int(np.sum(~above[:-1] & above[1:]))
    assert crossings == sweep.num_spikes
    assert -95 < np.median(tr.v[tr.t < 0]) < -60
