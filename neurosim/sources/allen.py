"""Client for the Allen Cell Types Database (https://celltypes.brain-map.org).

The Allen Institute recorded thousands of mouse and human neurons with
patch-clamp electrophysiology and reconstructed many of them. For each cell
this client exposes:

* metadata and Allen's own summary electrophysiology features
  (resting potential, input resistance, time constant, rheobase, ...),
* the list of recorded sweeps (stimulus type/amplitude and spike count),
* the morphology reconstruction (SWC),
* individual voltage traces, streamed from the NWB file with HTTP range
  requests so only the requested sweep (~2-4 MB) is downloaded instead of the
  whole 15-200 MB file.
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests

API_URL = "https://api.brain-map.org/api/v2/data/query.json"
FILE_URL = "https://api.brain-map.org/api/v2/well_known_file_download"
DEFAULT_CACHE = Path.home() / ".cache" / "neurosim" / "allen"
CELL_LIST_TTL = 7 * 24 * 3600  # seconds

SPECIES = {"Mus musculus": "mouse", "Homo Sapiens": "human"}


@dataclass
class AllenCell:
    specimen_id: int
    name: str
    species: str  # "mouse" | "human"
    region: str  # e.g. "VISp"
    region_name: str
    layer: str
    dendrite_type: str  # spiny (mostly excitatory) | aspiny (mostly inhibitory) | sparsely spiny
    line: str  # transgenic Cre line (mouse) or "" (human)
    reconstruction: str | None  # "full" | "dendrite-only" | None
    swc_file_id: int | None
    nwb_file_id: int | None
    features: dict[str, float] = field(default_factory=dict)  # Allen "ef__" summary features
    raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_api(cls, d: dict) -> "AllenCell":
        return cls(
            specimen_id=int(d["specimen__id"]),
            name=d.get("specimen__name") or "",
            species=SPECIES.get(d.get("donor__species"), d.get("donor__species") or ""),
            region=d.get("structure__acronym") or "",
            region_name=(d.get("structure__name") or "").strip('"'),
            layer=str(d.get("structure__layer") or ""),
            dendrite_type=d.get("tag__dendrite_type") or "",
            line=d.get("line_name") or "",
            reconstruction=d.get("nr__reconstruction_type"),
            swc_file_id=d.get("nrwkf__id"),
            nwb_file_id=d.get("erwkf__id"),
            features={k[4:]: v for k, v in d.items() if k.startswith("ef__") and v is not None},
            raw=d,
        )

    @property
    def page_url(self) -> str:
        return f"https://celltypes.brain-map.org/experiment/electrophysiology/{self.specimen_id}"

    @property
    def label(self) -> str:
        parts = [self.species, f"{self.region} L{self.layer}" if self.layer else self.region,
                 self.dendrite_type, self.line]
        return " · ".join(p for p in parts if p)


@dataclass
class Sweep:
    number: int
    stimulus: str  # e.g. "Long Square", "Short Square", "Ramp", "Noise 1"
    amplitude_pa: float
    start_s: float | None  # stimulus onset within the sweep
    duration_s: float | None
    num_spikes: int

    @classmethod
    def from_api(cls, d: dict) -> "Sweep":
        return cls(
            number=int(d["sweep_number"]),
            stimulus=d.get("stimulus_name") or "",
            amplitude_pa=float(d.get("stimulus_absolute_amplitude") or 0.0),
            start_s=d.get("stimulus_start_time"),
            duration_s=d.get("stimulus_duration"),
            num_spikes=int(d.get("num_spikes") or 0),  # the API reports None for 0
        )


@dataclass
class Trace:
    """A recorded voltage trace, with time in ms relative to stimulus onset."""

    t: np.ndarray  # ms (0 = stimulus onset)
    v: np.ndarray  # mV
    amplitude_pa: float
    duration_ms: float
    sweep: int


class _RangeFile(io.RawIOBase):
    """Read-only, seekable file over HTTP range requests with a block cache (for h5py)."""

    def __init__(self, url: str, session: requests.Session, block_size: int = 1 << 18, timeout: float = 60):
        self.session, self.block_size, self.timeout = session, block_size, timeout
        r = session.get(url, headers={"Range": "bytes=0-0"}, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        if r.status_code != 206:
            raise RuntimeError("server does not support range requests")
        self.url = r.url
        self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])
        self.pos = 0
        self._blocks: dict[int, bytes] = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def _block(self, i: int) -> bytes:
        if i not in self._blocks:
            a = i * self.block_size
            b = min(a + self.block_size, self.size) - 1
            r = self.session.get(self.url, headers={"Range": f"bytes={a}-{b}"}, timeout=self.timeout)
            r.raise_for_status()
            self._blocks[i] = r.content
        return self._blocks[i]

    def readinto(self, buf) -> int:
        n = min(len(buf), self.size - self.pos)
        if n <= 0:
            return 0
        out = bytearray()
        p = self.pos
        while len(out) < n:
            i = p // self.block_size
            off = p - i * self.block_size
            chunk = self._block(i)[off: off + n - len(out)]
            if not chunk:
                break
            out += chunk
            p += len(chunk)
        buf[: len(out)] = out
        self.pos += len(out)
        return len(out)


class AllenCellTypesClient:
    """Cached access to Allen Cell Types metadata, morphologies and recordings.

    >>> c = AllenCellTypesClient()
    >>> cells = c.cells(species="mouse", reconstructed=True)
    >>> sweeps = c.sweeps(cells[0].specimen_id)
    >>> trace = c.trace(cells[0], next(s for s in sweeps if s.stimulus == "Long Square" and s.num_spikes))
    """

    def __init__(self, cache_dir: Path | str | None = DEFAULT_CACHE, timeout: float = 60.0):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "neurosim/0.1 (+https://github.com/cdjcodes/neuroscience)"

    # --- helpers ------------------------------------------------------------
    def _query(self, criteria: str) -> list[dict]:
        rows, start, page = [], 0, 2000
        while True:
            r = self.session.get(
                API_URL, timeout=self.timeout,
                params={"criteria": f"{criteria},rma::options[start_row$eq{start}][num_rows$eq{page}]"},
            )
            r.raise_for_status()
            data = r.json()
            if not data.get("success", True):
                raise RuntimeError(f"Allen API error: {data.get('msg')}")
            rows += data["msg"]
            start += page
            if start >= int(data.get("total_rows", 0)):
                return rows

    def _cache(self, name: str) -> Path | None:
        return self.cache_dir / name if self.cache_dir else None

    # --- cells --------------------------------------------------------------
    def all_cells(self) -> list[AllenCell]:
        """Every cell in the database (~2.3k), cached on disk for a week."""
        path = self._cache("cells.json")
        if path and path.exists() and time.time() - path.stat().st_mtime < CELL_LIST_TTL:
            rows = json.loads(path.read_text())
        else:
            rows = self._query("model::ApiCellTypesSpecimenDetail")
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(rows))
        return [AllenCell.from_api(r) for r in rows]

    def cells(
        self,
        species: str | None = None,
        reconstructed: bool = True,
        dendrite_type: str | None = None,
        region: str | None = None,
    ) -> list[AllenCell]:
        """Filter the cell list. ``reconstructed=True`` keeps cells with a morphology."""
        out = []
        for c in self.all_cells():
            if reconstructed and not (c.reconstruction and c.swc_file_id):
                continue
            if species and c.species != species:
                continue
            if dendrite_type and c.dendrite_type != dendrite_type:
                continue
            if region and c.region != region:
                continue
            if not c.nwb_file_id:
                continue
            out.append(c)
        return out

    def get(self, specimen_id: int) -> AllenCell:
        for c in self.all_cells():
            if c.specimen_id == int(specimen_id):
                return c
        raise KeyError(f"No Allen cell with specimen id {specimen_id}")

    # --- morphology ---------------------------------------------------------
    def fetch_swc(self, cell: AllenCell | int) -> str:
        cell = cell if isinstance(cell, AllenCell) else self.get(cell)
        if not cell.swc_file_id:
            raise ValueError(f"cell {cell.specimen_id} has no reconstruction")
        path = self._cache(f"swc/{cell.specimen_id}.swc")
        if path and path.exists():
            return path.read_text()
        r = self.session.get(f"{FILE_URL}/{cell.swc_file_id}", timeout=self.timeout)
        r.raise_for_status()
        text = r.text.replace("\r\n", "\n")
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return text

    # --- electrophysiology --------------------------------------------------
    def sweeps(self, specimen_id: int) -> list[Sweep]:
        path = self._cache(f"sweeps/{specimen_id}.json")
        if path and path.exists():
            rows = json.loads(path.read_text())
        else:
            rows = self._query(f"model::EphysSweep,rma::criteria,[specimen_id$eq{int(specimen_id)}]")
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(rows))
        return sorted((Sweep.from_api(r) for r in rows), key=lambda s: s.number)

    @staticmethod
    def fi_curve(sweeps: list[Sweep]) -> tuple[np.ndarray, np.ndarray]:
        """Measured f-I curve from the long-square sweeps: (current in nA, rate in Hz).

        Repeated amplitudes are averaged.
        """
        by_amp: dict[float, list[float]] = {}
        for s in sweeps:
            if s.stimulus == "Long Square" and s.amplitude_pa >= 0 and s.duration_s:
                by_amp.setdefault(round(s.amplitude_pa), []).append(s.num_spikes / s.duration_s)
        amps = np.array(sorted(by_amp), dtype=float)
        rates = np.array([np.mean(by_amp[a]) for a in sorted(by_amp)])
        return amps / 1000.0, rates

    def trace(
        self,
        cell: AllenCell | int,
        sweep: Sweep,
        pre_ms: float = 100.0,
        post_ms: float = 200.0,
        sample_rate_hz: float = 50_000.0,
    ) -> Trace:
        """Voltage of one sweep around its stimulus, streamed from the NWB file.

        Only ``[onset - pre_ms, offset + post_ms]`` is kept, downsampled to
        ``sample_rate_hz``. Results are cached on disk.
        """
        import h5py  # optional dependency, only needed for traces

        cell = cell if isinstance(cell, AllenCell) else self.get(cell)
        if sweep.start_s is None or sweep.duration_s is None:
            raise ValueError("sweep has no stimulus timing")
        dur_ms = sweep.duration_s * 1000.0
        path = self._cache(f"traces/{cell.specimen_id}_{sweep.number}_{int(pre_ms)}_{int(post_ms)}.npz")
        if path and path.exists():
            d = np.load(path)
            return Trace(d["t"], d["v"], sweep.amplitude_pa, dur_ms, sweep.number)

        f = _RangeFile(f"{FILE_URL}/{cell.nwb_file_id}", self.session, timeout=self.timeout)
        try:
            with h5py.File(f, "r") as h5:
                ts = h5[f"acquisition/timeseries/Sweep_{sweep.number}"]
                rate = float(ts["starting_time"].attrs["rate"])
                step = max(1, int(round(rate / sample_rate_hz)))
                a = max(0, int((sweep.start_s - pre_ms / 1000) * rate))
                b = int((sweep.start_s + sweep.duration_s + post_ms / 1000) * rate)
                v = np.asarray(ts["data"][a:b:step], dtype=float)
                conv = float(ts["data"].attrs.get("conversion", 1.0))
        finally:
            f.close()
        v = v * conv * 1000.0  # V -> mV
        t = (a + np.arange(len(v)) * step) / rate * 1000.0 - sweep.start_s * 1000.0
        ok = np.isfinite(v)
        t, v = t[ok], v[ok]
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, t=t, v=v)
        return Trace(t, v, sweep.amplitude_pa, dur_ms, sweep.number)
