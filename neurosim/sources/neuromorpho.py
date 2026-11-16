"""Client for the NeuroMorpho.org REST API (https://neuromorpho.org/apiReference.html).

NeuroMorpho.org hosts over 200k digitally reconstructed neurons contributed by
labs worldwide. Each neuron has searchable metadata (species, brain region,
cell type, etc.) and a standardised SWC reconstruction ("CNG version").
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

import requests

API_URL = "https://neuromorpho.org/api"
FILES_URL = "https://neuromorpho.org/dableFiles"
DEFAULT_CACHE = Path.home() / ".cache" / "neurosim" / "neuromorpho"


@dataclass
class NeuronRecord:
    """Metadata for one reconstructed neuron on NeuroMorpho.org."""

    neuron_id: int
    name: str
    archive: str
    species: str
    brain_region: list[str] = field(default_factory=list)
    cell_type: list[str] = field(default_factory=list)
    domain: str = ""
    png_url: str = ""
    reference_doi: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_api(cls, d: dict) -> "NeuronRecord":
        return cls(
            neuron_id=d["neuron_id"],
            name=d["neuron_name"],
            archive=d["archive"],
            species=d.get("species", ""),
            brain_region=d.get("brain_region") or [],
            cell_type=d.get("cell_type") or [],
            domain=d.get("domain", ""),
            png_url=(d.get("png_url") or "").replace("http://", "https://", 1),
            reference_doi=d.get("reference_doi") or [],
            raw=d,
        )

    @property
    def swc_url(self) -> str:
        """URL of the standardised (CNG) SWC reconstruction."""
        archive = quote(self.archive.lower())
        return f"{FILES_URL}/{archive}/CNG%20version/{quote(self.name)}.CNG.swc"

    @property
    def page_url(self) -> str:
        return f"https://neuromorpho.org/neuron_info.jsp?neuron_name={quote(self.name)}"


@dataclass
class SearchResult:
    neurons: list[NeuronRecord]
    total: int
    page: int
    total_pages: int


class NeuroMorphoClient:
    """Thin, cached wrapper around the NeuroMorpho.org API.

    >>> client = NeuroMorphoClient()
    >>> res = client.search(species="mouse", cell_type="pyramidal", size=5)
    >>> swc_path = client.download_swc(res.neurons[0])
    """

    def __init__(self, cache_dir: Path | str | None = DEFAULT_CACHE, timeout: float = 30.0):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "neurosim/0.1 (+https://github.com/cdjcodes/neuroscience)"

    def _get(self, path: str, params=None) -> requests.Response:
        return self.session.get(f"{API_URL}/{path}", params=params, timeout=self.timeout)

    # --- metadata -----------------------------------------------------------
    def field_values(self, field_name: str) -> list[str]:
        """All values a metadata field takes, e.g. ``field_values("species")``."""
        resp = self._get(f"neuron/fields/{field_name}")
        resp.raise_for_status()
        return resp.json()["fields"]

    def search(
        self, page: int = 0, size: int = 20, **filters: str | list[str] | None
    ) -> SearchResult:
        """Search neurons by metadata, e.g. ``search(species="rat", brain_region="hippocampus")``.

        Any NeuroMorpho metadata field can be used as a keyword filter. A list
        value matches any of its entries. Empty/None filters are ignored; with
        no filters every neuron matches.
        """
        body = {k: [v] if isinstance(v, str) else list(v) for k, v in filters.items() if v}
        params = {"page": page, "size": min(size, 500)}
        if body:
            resp = self.session.post(
                f"{API_URL}/neuron/select", params=params, json=body, timeout=self.timeout
            )
        else:
            resp = self._get("neuron/select", params={**params, "q": "domain:*"})
        if resp.status_code == 404:  # API answers 404 for "no matches"
            return SearchResult([], 0, page, 0)
        resp.raise_for_status()
        data = resp.json()
        neurons = [NeuronRecord.from_api(d) for d in data["_embedded"]["neuronResources"]]
        p = data["page"]
        return SearchResult(neurons, p["totalElements"], p["number"], p["totalPages"])

    def get(self, name: str) -> NeuronRecord:
        """Look up a single neuron by its NeuroMorpho name."""
        resp = self._get(f"neuron/name/{quote(name)}")
        resp.raise_for_status()
        return NeuronRecord.from_api(resp.json())

    def morphometry(self, neuron_id: int) -> dict:
        """NeuroMorpho's own pre-computed morphometrics (L-Measure) for a neuron."""
        resp = self._get(f"morphometry/id/{neuron_id}")
        resp.raise_for_status()
        return resp.json()

    # --- files --------------------------------------------------------------
    def fetch_swc(self, neuron: NeuronRecord | str) -> str:
        """Return the SWC reconstruction as text (cached on disk when possible)."""
        if isinstance(neuron, str):
            neuron = self.get(neuron)
        cached = self._cache_path(neuron)
        if cached and cached.exists():
            return cached.read_text()
        resp = self.session.get(neuron.swc_url, timeout=self.timeout)
        resp.raise_for_status()
        text = resp.text.replace("\r\n", "\n")
        if "<html" in text[:500].lower():
            raise RuntimeError(f"NeuroMorpho returned HTML instead of SWC for {neuron.name}")
        if cached:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(text)
        return text

    def download_swc(self, neuron: NeuronRecord | str, dest: Path | str | None = None) -> Path:
        """Download the SWC file and return its path."""
        if isinstance(neuron, str):
            neuron = self.get(neuron)
        text = self.fetch_swc(neuron)
        path = Path(dest) if dest else self._cache_path(neuron)
        if path is None:
            raise ValueError("No cache_dir configured; pass dest explicitly")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return path

    def _cache_path(self, neuron: NeuronRecord) -> Path | None:
        if not self.cache_dir:
            return None
        # names can contain characters that are awkward in filenames
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in neuron.name)
        digest = hashlib.sha1(neuron.name.encode()).hexdigest()[:8]
        return self.cache_dir / f"{safe}-{digest}.swc"
