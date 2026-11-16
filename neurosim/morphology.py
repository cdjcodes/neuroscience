"""Neuron morphology loading and morphometrics, built on NeuroM
(https://github.com/openbraininstitute/NeuroM)."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import morphio
import neurom as nm
import numpy as np
from neurom import NeuriteType
from neurom.core.morphology import Morphology

NEURITE_TYPES = {
    "axon": NeuriteType.axon,
    "basal_dendrite": NeuriteType.basal_dendrite,
    "apical_dendrite": NeuriteType.apical_dendrite,
}

# Database files trigger many harmless MorphIO warnings; keep the output clean.
morphio.set_maximum_warnings(0)


def load(source: str | Path, fmt: str = "swc", repair: bool = True) -> Morphology:
    """Load a morphology from a file path or from SWC/ASC text.

    Strings containing newlines are treated as file contents, so the text
    returned by :meth:`NeuroMorphoClient.fetch_swc` can be passed directly.

    Many database reconstructions have somas that MorphIO rejects (branched
    soma contours, missing soma, several roots). With ``repair=True`` such SWC
    files are retried after :func:`normalize_swc_soma`.
    """
    is_text = isinstance(source, str) and "\n" in source
    try:
        if is_text:
            morph = nm.load_morphology(io.StringIO(source), reader=fmt)
        else:
            morph = nm.load_morphology(source)
        if len(morph.soma.points) == 0:
            raise ValueError("morphology has no soma")
        return morph
    except Exception:
        if not repair or fmt != "swc":
            raise
        text = source if is_text else Path(source).read_text()
        return nm.load_morphology(io.StringIO(normalize_swc_soma(text)), reader="swc")


def normalize_swc_soma(text: str) -> str:
    """Collapse all soma samples of an SWC file into a single-point soma.

    Children of any soma sample (and any extra tree roots) are re-attached to
    the new soma. If the file has no soma, one is created at the first root.
    """
    rows, comments = [], []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            comments.append(line)
            continue
        f = s.split()
        rows.append([int(float(f[0])), int(float(f[1])), *map(float, f[2:6]), int(float(f[6]))])

    soma = [r for r in rows if r[1] == 1]
    if soma:
        xyz = np.array([r[2:5] for r in soma])
        center = xyz.mean(axis=0)
        radius = max(np.mean([r[5] for r in soma]), np.linalg.norm(xyz - center, axis=1).mean())
    else:
        root = next(r for r in rows if r[6] == -1)
        center, radius = np.array(root[2:5]), max(root[5], 1.0)
    radius = float(radius) or 1.0

    soma_ids = {r[0] for r in soma}
    new_id = max(r[0] for r in rows) + 1
    out = [f"{new_id} 1 {center[0]:.4f} {center[1]:.4f} {center[2]:.4f} {radius:.4f} -1"]
    for r in rows:
        if r[1] == 1:
            continue
        parent = new_id if (r[6] in soma_ids or r[6] == -1) else r[6]
        out.append(f"{r[0]} {r[1]} {r[2]:.4f} {r[3]:.4f} {r[4]:.4f} {r[5]:.4f} {parent}")
    return "\n".join(comments + out) + "\n"


def _get(feature: str, morph: Morphology, **kw) -> float:
    try:
        value = nm.get(feature, morph, **kw)
    except Exception:  # feature undefined for this neuron, e.g. no axon
        return float("nan")
    return float(np.sum(value)) if np.ndim(value) else float(value)


def summary(morph: Morphology) -> dict[str, float]:
    """Key morphometrics for the whole neuron (lengths in µm, areas µm², volumes µm³)."""
    return {
        "soma_radius": float(morph.soma.radius),
        "number_of_neurites": _get("number_of_neurites", morph),
        "number_of_sections": _get("number_of_sections", morph),
        "number_of_bifurcations": _get("number_of_bifurcations", morph),
        "number_of_leaves": _get("number_of_leaves", morph),
        "total_length": _get("total_length", morph),
        "total_area": _get("total_area", morph),
        "total_volume": _get("total_volume", morph),
        "max_radial_distance": _get("max_radial_distance", morph),
    }


def per_neurite_type(morph: Morphology) -> dict[str, dict[str, float]]:
    """Morphometrics broken down by neurite type (only types present are returned)."""
    out = {}
    for name, ntype in NEURITE_TYPES.items():
        n = _get("number_of_neurites", morph, neurite_type=ntype)
        if not n:
            continue
        out[name] = {
            "number_of_neurites": n,
            "number_of_sections": _get("number_of_sections", morph, neurite_type=ntype),
            "number_of_bifurcations": _get("number_of_bifurcations", morph, neurite_type=ntype),
            "total_length": _get("total_length", morph, neurite_type=ntype),
            "max_radial_distance": _get("max_radial_distance", morph, neurite_type=ntype),
        }
    return out


def sholl(morph: Morphology, step_size: float = 10.0, neurite_type=NeuriteType.all):
    """Sholl analysis: number of intersections with concentric spheres around the soma.

    Returns ``(radii, intersections)`` arrays.
    """
    freq = np.asarray(nm.get("sholl_frequency", morph, step_size=step_size, neurite_type=neurite_type))
    radii = np.arange(len(freq)) * step_size
    return radii, freq


def distributions(morph: Morphology) -> dict[str, np.ndarray]:
    """Per-section / per-bifurcation distributions useful for histograms."""
    keys = {
        "section_lengths": "section_lengths",
        "section_path_distances": "section_path_distances",
        "remote_bifurcation_angles": "remote_bifurcation_angles",
        "section_branch_orders": "section_branch_orders",
    }
    return {k: np.asarray(nm.get(f, morph), dtype=float) for k, f in keys.items()}


@dataclass
class Geometry:
    """Line-segment geometry of a neuron, ready for plotting.

    ``lines[type]`` is an (N, 3) array of points where separate sections are
    separated by a row of NaNs (the format plotly/matplotlib expect for
    disconnected polylines). ``widths[type]`` holds the matching diameters and
    ``section_index[type]`` the index (in :func:`neurom.iter_sections` order,
    the same order used by :mod:`neurosim.simulation`) of the section each
    point belongs to, or -1 for separators.
    """

    soma_center: np.ndarray
    soma_radius: float
    lines: dict[str, np.ndarray]
    widths: dict[str, np.ndarray]
    section_index: dict[str, np.ndarray]


def geometry(morph: Morphology) -> Geometry:
    lines: dict[str, list] = {}
    widths: dict[str, list] = {}
    index: dict[str, list] = {}
    for i, section in enumerate(nm.iter_sections(morph)):
        name = section.type.name
        pts = section.points
        lines.setdefault(name, []).extend([pts[:, :3], np.full((1, 3), np.nan)])
        widths.setdefault(name, []).extend([2 * pts[:, 3], [np.nan]])
        index.setdefault(name, []).extend([np.full(len(pts), i), [-1]])
    return Geometry(
        soma_center=np.asarray(morph.soma.center, dtype=float),
        soma_radius=float(morph.soma.radius),
        lines={k: np.vstack(v) for k, v in lines.items()},
        widths={k: np.concatenate(v) for k, v in widths.items()},
        section_index={k: np.concatenate(v).astype(int) for k, v in index.items()},
    )
