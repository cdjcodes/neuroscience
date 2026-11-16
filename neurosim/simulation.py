"""Current-clamp experiments with NEURON (https://nrn.readthedocs.io).

Three kinds of cell are provided:

* :class:`BallAndStick` – the classic teaching model from the EPFL
  *Simulation Neuroscience* course: a Hodgkin-Huxley soma and a passive
  dendrite.
* :class:`MorphologyCell` – a multi-compartment model built from a real
  reconstructed neuron (e.g. one downloaded from NeuroMorpho.org), with
  Hodgkin-Huxley channels in the soma/axon and passive (or optionally
  active) dendrites.
* :class:`AllenPerisomaticCell` – a real morphology with the Allen
  Institute's published, cell-specific "perisomatic" biophysical model:
  cortical Na, K, Ca and Ih channels (compiled from
  ``neurosim/mechanisms``) fitted to that cell's recordings.

All share :meth:`Cell.run` (single current step) and :func:`fi_curve`.

NEURON keeps one global simulation state, so all building/running is
serialised through a lock, which makes the module safe to use from a
multi-threaded web server.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field, replace

import neurom as nm
import numpy as np
from neuron import h

h.load_file("stdrun.hoc")
_LOCK = threading.RLock()


@dataclass
class CurrentClamp:
    """A square current pulse injected into the soma."""

    amp: float = 0.5  # nA
    delay: float = 20.0  # ms
    dur: float = 100.0  # ms
    tstop: float = 150.0  # ms
    dt: float = 0.025  # ms


@dataclass
class Biophysics:
    """Passive and active membrane parameters."""

    Ra: float = 150.0  # axial resistivity, ohm*cm
    cm: float = 1.0  # membrane capacitance, uF/cm2
    # Dendritic capacitance (MorphologyCell). Spines add membrane that
    # reconstructions omit, which shows up as a higher apparent capacitance in
    # dendrites only. None = same as ``cm``.
    cm_dendrite: float | None = None
    celsius: float = 6.3  # temperature the HH model was fitted at
    # Hodgkin-Huxley (soma + axon), S/cm2
    gnabar: float = 0.12
    gkbar: float = 0.036
    gl: float = 0.0003
    el: float = -54.3  # mV
    # Passive dendrites
    g_pas: float = 3e-5  # S/cm2  (Rm ~ 33 kOhm*cm2)
    e_pas: float = -65.0  # mV
    # Scale HH Na/K density in dendrites (0 = passive dendrites)
    dendrite_hh_scale: float = 0.0
    v_init: float = -65.0  # mV
    # MorphologyCell only. A large passive dendritic tree loads a squid-HH
    # soma so heavily that it fires a single spike and then blocks. Real
    # neurons initiate spikes in the axon initial segment (AIS), where Na
    # channel density is much higher, so we model one: the first
    # ``ais_length`` um of the reconstructed axon, or a synthetic stub if the
    # reconstruction has no axon.
    soma_hh_scale: float = 3.0
    ais_hh_scale: float = 20.0
    ais_length: float = 40.0  # um
    ais_diam: float = 1.0  # um, synthetic AIS only
    # If True the HH leak is switched off and one passive leak (g_pas, e_pas)
    # is used in every section, so resting potential, input resistance and
    # time constant are set by e_pas, g_pas and cm alone. Used by fitting.
    uniform_leak: bool = False
    # Spatial discretisation (d_lambda rule); larger = fewer compartments, faster.
    d_lambda: float = 0.1


@dataclass
class SimulationResult:
    t: np.ndarray  # ms
    v: dict[str, np.ndarray]  # recording site -> membrane potential (mV)
    i_inj: np.ndarray  # injected current (nA), sampled on t
    spike_times: np.ndarray  # soma spike times (ms)
    stim: CurrentClamp
    # Optional per-section voltage, shape (n_sections, n_samples), sampled on t_sections
    v_sections: np.ndarray | None = None
    t_sections: np.ndarray | None = None
    info: dict = field(default_factory=dict)

    @property
    def firing_rate(self) -> float:
        """Mean firing rate during the stimulus (Hz)."""
        s = self.stim
        n = np.sum((self.spike_times >= s.delay) & (self.spike_times <= s.delay + s.dur))
        return 1000.0 * n / s.dur if s.dur > 0 else 0.0


def detect_spikes(t: np.ndarray, v: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Times at which ``v`` crosses ``threshold`` upwards."""
    above = v >= threshold
    idx = np.flatnonzero(~above[:-1] & above[1:]) + 1
    return t[idx]


def _set_nseg(sec, d_lambda: float = 0.1, freq: float = 100.0) -> None:
    """d_lambda rule: segment length <= d_lambda * AC length constant at ``freq``."""
    lam = 1e5 * math.sqrt(sec.diam / (4 * math.pi * freq * sec.Ra * sec.cm))
    sec.nseg = int((sec.L / (d_lambda * lam) + 0.9) / 2) * 2 + 1


def _insert_hh(sec, bio: Biophysics, scale: float = 1.0) -> None:
    sec.insert("hh")
    for seg in sec:
        seg.hh.gnabar = bio.gnabar * scale
        seg.hh.gkbar = bio.gkbar * scale
        seg.hh.gl = 0.0 if bio.uniform_leak else bio.gl
        seg.hh.el = bio.el
    if bio.uniform_leak:
        _insert_pas(sec, bio)


def _insert_pas(sec, bio: Biophysics) -> None:
    sec.insert("pas")
    for seg in sec:
        seg.pas.g = bio.g_pas
        seg.pas.e = bio.e_pas


class Cell:
    """Base class: subclasses create ``self.soma`` and fill ``self.sites``."""

    def __init__(self, bio: Biophysics | None = None):
        self.bio = bio or Biophysics()
        self.soma = None
        self.sections: list = []  # all non-soma sections
        self.sites: dict[str, object] = {}  # name -> segment to record from
        # Added to every recorded voltage, e.g. to express a model fitted to
        # junction-potential-corrected data in the recording's frame.
        self.v_offset = 0.0

    def run(self, stim: CurrentClamp | None = None, record_sections_dt: float | None = None) -> SimulationResult:
        """Inject a current step into the soma and record the response.

        If ``record_sections_dt`` is given, the voltage at the midpoint of
        every section in ``self.sections`` is also recorded, sampled every
        ``record_sections_dt`` ms (useful for animating the morphology).
        """
        stim = stim or CurrentClamp()
        with _LOCK:
            clamp = h.IClamp(self.soma(0.5))
            clamp.amp, clamp.delay, clamp.dur = stim.amp, stim.delay, stim.dur

            t_vec = h.Vector().record(h._ref_t)
            i_vec = h.Vector().record(clamp._ref_i)
            site_vecs = {name: h.Vector().record(seg._ref_v) for name, seg in self.sites.items()}
            sec_vecs = []
            if record_sections_dt:
                t_sec_vec = h.Vector().record(h._ref_t, record_sections_dt)
                sec_vecs = [h.Vector().record(s(0.5)._ref_v, record_sections_dt) for s in self.sections]

            h.celsius = self.bio.celsius
            h.dt = stim.dt
            h.finitialize(self.bio.v_init)
            h.continuerun(stim.tstop)

            t = t_vec.as_numpy().copy()
            v = {name: vec.as_numpy() + self.v_offset for name, vec in site_vecs.items()}
            result = SimulationResult(
                t=t,
                v=v,
                i_inj=i_vec.as_numpy().copy(),
                spike_times=detect_spikes(t, v["soma"]),
                stim=stim,
            )
            if record_sections_dt:
                result.t_sections = t_sec_vec.as_numpy().copy()
                result.v_sections = (np.array([vec.as_numpy() for vec in sec_vecs]) + self.v_offset
                                     if sec_vecs else None)
            del clamp
        return result


class BallAndStick(Cell):
    """HH soma + passive dendrite, as in the course's ball-and-stick notebooks."""

    def __init__(
        self,
        soma_diam: float = 20.0,
        dend_length: float = 200.0,
        dend_diam: float = 1.0,
        bio: Biophysics | None = None,
    ):
        super().__init__(bio)
        with _LOCK:
            self.soma = h.Section(name="soma")
            self.soma.L = self.soma.diam = soma_diam
            self.dend = h.Section(name="dend")
            self.dend.L, self.dend.diam = dend_length, dend_diam
            self.dend.connect(self.soma(1))
            for sec in (self.soma, self.dend):
                sec.Ra, sec.cm = self.bio.Ra, self.bio.cm
            _insert_hh(self.soma, self.bio)
            _insert_pas(self.dend, self.bio)
            if self.bio.dendrite_hh_scale > 0:
                _insert_hh(self.dend, self.bio, self.bio.dendrite_hh_scale)
            self.dend.nseg = 51
            self.sections = [self.dend]
            self.sites = {"soma": self.soma(0.5), "dendrite_mid": self.dend(0.5), "dendrite_tip": self.dend(1.0)}


class MorphologyCell(Cell):
    """A multi-compartment cell built from a NeuroM morphology.

    Sections are created in :func:`neurom.iter_sections` order, so
    ``self.sections[i]`` matches ``Geometry.section_index == i`` from
    :func:`neurosim.morphology.geometry`.
    """

    def __init__(self, morph, bio: Biophysics | None = None):
        super().__init__(bio)
        b = self.bio
        with _LOCK:
            self.soma = h.Section(name="soma")
            # Represent the soma as a cylinder with the same area as a sphere of radius r.
            r = max(float(morph.soma.radius), 1.0)
            self.soma.L = self.soma.diam = 2 * r
            self.soma.Ra, self.soma.cm = b.Ra, b.cm
            _insert_hh(self.soma, b, b.soma_hh_scale)

            by_id: dict[int, object] = {}
            for i, s in enumerate(nm.iter_sections(morph)):
                sec = h.Section(name=f"{s.type.name}_{i}")
                for x, y, z, rad in s.points[:, :4]:
                    sec.pt3dadd(float(x), float(y), float(z), max(2 * float(rad), 0.1))
                parent = by_id.get(s.parent.id) if s.parent is not None else None
                sec.connect(parent(1) if parent is not None else self.soma(0.5))
                is_axon = s.type == nm.NeuriteType.axon
                sec.Ra = b.Ra
                sec.cm = b.cm if is_axon or b.cm_dendrite is None else b.cm_dendrite
                _set_nseg(sec, b.d_lambda)
                if is_axon:
                    _insert_hh(sec, b)
                else:
                    _insert_pas(sec, b)
                    if b.dendrite_hh_scale > 0:
                        _insert_hh(sec, b, b.dendrite_hh_scale)
                by_id[s.id] = sec
                self.sections.append(sec)

            self._add_ais(morph)
            self.sites = {"soma": self.soma(0.5)}
            # Also record at the dendritic tip farthest (by path) from the soma.
            dends = [sec for sec in self.sections if "dendrite" in sec.name()]
            if dends:
                h.distance(0, self.soma(0.5))
                tip = max(dends, key=lambda sec: h.distance(sec(1)))
                self.sites["farthest_dendrite"] = tip(1)
                self.farthest_dendrite_um = h.distance(tip(1))
            axons = [sec for sec in self.sections if sec.name().startswith("axon")]
            if axons:
                self.sites["axon"] = axons[0](1)

    def _add_ais(self, morph) -> None:
        b = self.bio
        self.ais = None
        if b.ais_hh_scale <= 0 or b.ais_length <= 0:
            return
        h.distance(0, self.soma(0.5))
        axons = [sec for sec in self.sections if sec.name().startswith("axon")]
        if not axons:
            self.ais = h.Section(name="ais")
            self.ais.L, self.ais.diam = b.ais_length, b.ais_diam
            self.ais.Ra, self.ais.cm = b.Ra, b.cm
            self.ais.nseg = 9
            self.ais.connect(self.soma(0.5))
            _insert_hh(self.ais, b, b.ais_hh_scale)
            return
        for sec in axons:
            for seg in sec:
                if h.distance(seg) <= b.ais_length:
                    seg.hh.gnabar = b.gnabar * b.ais_hh_scale
                    seg.hh.gkbar = b.gkbar * b.ais_hh_scale

    @property
    def n_compartments(self) -> int:
        extra = self.ais.nseg if self.ais is not None else 0
        return self.soma.nseg + extra + sum(sec.nseg for sec in self.sections)


class AllenPerisomaticCell(Cell):
    """A real morphology with an Allen Institute "Biophysical - perisomatic" model.

    ``params`` is the model's fit file (see
    :meth:`AllenCellTypesClient.perisomatic_model`). Construction follows the
    Allen recipe: dendrites from the reconstruction, the axon replaced by a
    60 um stub, ``nseg = 1 + 2 * int(L / 40)``, per-section-type passive
    properties, active channels from the fitted "genome" (mostly somatic),
    and the fit's temperature, reversal potentials and initial voltage.

    Allen fitted these models to recordings corrected for a liquid junction
    potential, so recorded voltages are shifted back into the uncorrected
    frame of the raw recordings (``v_offset = -junction_potential``).

    ``self.sections[i]`` matches ``Geometry.section_index == i``; axon
    geometry maps to the axon stub.
    """

    def __init__(self, morph, params: dict):
        from .channels import ensure_perisomatic_channels

        ensure_perisomatic_channels()
        passive = params["passive"][0]
        cond = params["conditions"][0]
        fitting = (params.get("fitting") or [{}])[0]
        super().__init__(replace(Biophysics(), celsius=float(cond["celsius"]), v_init=float(cond["v_init"])))
        self.v_offset = -float(fitting.get("junction_potential", 0.0))
        self.params = params

        with _LOCK:
            self.soma = h.Section(name="soma")
            r = max(float(morph.soma.radius), 1.0)
            self.soma.L = self.soma.diam = 2 * r
            groups: dict[str, list] = {"soma": [self.soma], "dend": [], "apic": [], "axon": []}

            by_id: dict[int, object] = {}
            geometry_sections: list = []
            for i, s in enumerate(nm.iter_sections(morph)):
                if s.type == nm.NeuriteType.axon:
                    geometry_sections.append(None)
                    continue
                kind = "apic" if s.type == nm.NeuriteType.apical_dendrite else "dend"
                sec = h.Section(name=f"{kind}_{i}")
                for x, y, z, rad in s.points[:, :4]:
                    sec.pt3dadd(float(x), float(y), float(z), max(2 * float(rad), 0.1))
                parent = by_id.get(s.parent.id) if s.parent is not None else None
                sec.connect(parent(1) if parent is not None else self.soma(0.5))
                groups[kind].append(sec)
                by_id[s.id] = sec
                geometry_sections.append(sec)

            # Allen replaces the reconstructed axon with a 60 um, 1 um-diameter stub.
            ax0, ax1 = h.Section(name="axon_0"), h.Section(name="axon_1")
            for ax in (ax0, ax1):
                ax.L, ax.diam = 30.0, 1.0
            ax0.connect(self.soma(0.5))
            ax1.connect(ax0(1))
            groups["axon"] = [ax0, ax1]
            self._owned = [sec for secs in groups.values() for sec in secs]

            cm = {c["section"]: float(c["cm"]) for c in passive["cm"]}
            for kind, secs in groups.items():
                for sec in secs:
                    sec.nseg = 1 + 2 * int(sec.L / 40.0)
                    sec.Ra = float(passive["ra"])
                    sec.cm = cm.get(kind, 1.0)
                    sec.insert("pas")
                    for seg in sec:
                        seg.pas.e = float(passive["e_pas"])
            for gene in params["genome"]:
                for sec in groups.get(gene["section"], []):
                    if gene["mechanism"]:
                        sec.insert(gene["mechanism"])
                    setattr(sec, gene["name"], float(gene["value"]))
            for erev in cond.get("erev", []):
                for sec in groups.get(erev["section"], []):
                    if "ena" in erev and h.ismembrane("na_ion", sec=sec):
                        sec.ena = float(erev["ena"])
                    if "ek" in erev and h.ismembrane("k_ion", sec=sec):
                        sec.ek = float(erev["ek"])

            self.sections = [sec if sec is not None else ax1 for sec in geometry_sections]
            self.ais = None
            self.sites = {"soma": self.soma(0.5)}
            dends = groups["dend"] + groups["apic"]
            if dends:
                h.distance(0, self.soma(0.5))
                tip = max(dends, key=lambda sec: h.distance(sec(1)))
                self.sites["farthest_dendrite"] = tip(1)
                self.farthest_dendrite_um = h.distance(tip(1))
            self.sites["axon"] = ax1(1)

    @property
    def n_compartments(self) -> int:
        return sum(sec.nseg for sec in self._owned)


def fi_curve(cell: Cell, amps, stim: CurrentClamp | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Frequency-current (f-I) curve: firing rate (Hz) for each injected amplitude (nA)."""
    base = stim or CurrentClamp()
    rates = []
    for a in amps:
        res = cell.run(CurrentClamp(a, base.delay, base.dur, base.tstop, base.dt))
        rates.append(res.firing_rate)
    return np.asarray(amps, dtype=float), np.asarray(rates)


def rheobase(cell: Cell, lo: float = 0.0, hi: float = 5.0, tol: float = 0.01,
             stim: CurrentClamp | None = None) -> float | None:
    """Smallest step current (nA) that evokes at least one spike, by bisection.

    Returns ``None`` if even ``hi`` does not make the cell fire.
    """
    base = stim or CurrentClamp()

    def fires(a: float) -> bool:
        return len(cell.run(CurrentClamp(a, base.delay, base.dur, base.tstop, base.dt)).spike_times) > 0

    if not fires(hi):
        return None
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if fires(mid) else (mid, hi)
    return hi
