"""Cortical ion-channel models for NEURON, compiled on demand.

The Allen Institute's "Biophysical - perisomatic" models use a set of 16
NMODL channel descriptions (transient/persistent Na, several K currents,
Ca currents and Ca dynamics, Ih). Reviewed copies are pinned in
``neurosim/mechanisms/allen_perisomatic`` (see the README there). On first
use they are copied to a cache directory, compiled with NEURON's
``nrnivmodl`` (needs a C/C++ compiler) and loaded into the running NEURON
process. Nothing is downloaded at runtime.
"""

from __future__ import annotations

import hashlib
import platform
import shutil
import subprocess
import sys
import threading
from pathlib import Path

SOURCE_DIR = Path(__file__).parent / "mechanisms" / "allen_perisomatic"
CACHE_ROOT = Path.home() / ".cache" / "neurosim" / "mechanisms"

_LOCK = threading.Lock()
_loaded: Path | None = None


class ChannelsUnavailable(RuntimeError):
    """The channel models could not be compiled or loaded."""


def _nrnivmodl() -> str:
    for c in (Path(sys.executable).parent / "nrnivmodl", shutil.which("nrnivmodl")):
        if c and Path(c).exists():
            return str(c)
    raise ChannelsUnavailable("nrnivmodl (part of the NEURON package) was not found")


def _digest(files: list[Path]) -> str:
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def _compiled_lib(build_dir: Path) -> Path | None:
    arch_dir = build_dir / platform.machine()
    for pattern in ("libnrnmech.*", ".libs/libnrnmech.*"):
        found = sorted(arch_dir.glob(pattern))
        if found:
            return found[0]
    return None


def ensure_perisomatic_channels() -> Path:
    """Compile (first time only, ≈10-60 s) and load the Allen perisomatic channels.

    Returns the build directory. Safe to call repeatedly and from several threads.
    """
    global _loaded
    with _LOCK:
        if _loaded is not None:
            return _loaded
        sources = sorted(SOURCE_DIR.glob("*.mod"))
        if not sources:
            raise ChannelsUnavailable(f"no .mod files found in {SOURCE_DIR}")
        # Build directory keyed by file contents, so edited files trigger a rebuild.
        build_dir = CACHE_ROOT / f"allen_perisomatic-{_digest(sources)}"
        if _compiled_lib(build_dir) is None:
            build_dir.mkdir(parents=True, exist_ok=True)
            for f in sources:
                shutil.copy2(f, build_dir / f.name)
            try:
                proc = subprocess.run([_nrnivmodl()], cwd=build_dir, capture_output=True, text=True, timeout=900)
            except (OSError, subprocess.TimeoutExpired) as e:
                raise ChannelsUnavailable(f"could not run nrnivmodl: {e}") from e
            if proc.returncode != 0 or _compiled_lib(build_dir) is None:
                tail = (proc.stderr or proc.stdout or "")[-1500:]
                raise ChannelsUnavailable(
                    f"compiling the channel models failed (a C/C++ compiler is required):\n{tail}")

        import neuron

        if not neuron.load_mechanisms(str(build_dir)):
            raise ChannelsUnavailable("NEURON could not load the compiled channel library")
        _loaded = build_dir
        return build_dir


def available() -> bool:
    """True if the perisomatic channels are loaded or can be compiled and loaded."""
    try:
        ensure_perisomatic_channels()
        return True
    except ChannelsUnavailable:
        return False
