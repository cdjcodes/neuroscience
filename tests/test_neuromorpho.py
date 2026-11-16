"""Live tests against NeuroMorpho.org. Skip with ``pytest -m 'not network'``."""
import pytest

from neurosim import morphology as mo
from neurosim.sources import NeuroMorphoClient
from neurosim.sources.neuromorpho import NeuronRecord

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    return NeuroMorphoClient(cache_dir=tmp_path_factory.mktemp("nm"))


def test_swc_url_escapes_archive_name():
    rec = NeuronRecord(1, "cell 1", "Allen Cell Types", "mouse")
    assert rec.swc_url.endswith("/allen%20cell%20types/CNG%20version/cell%201.CNG.swc")


def test_search_with_multiword_filters(client):
    res = client.search(species="mouse", archive="Allen Cell Types", size=3)
    assert res.total > 0 and len(res.neurons) == 3
    assert all(n.archive == "Allen Cell Types" for n in res.neurons)


def test_search_no_match(client):
    assert client.search(species="unicorn").total == 0


def test_fetch_and_analyse(client):
    text = client.fetch_swc("n419")
    assert client.fetch_swc("n419") == text  # served from cache
    s = mo.summary(mo.load(text))
    assert s["total_length"] > 1000
