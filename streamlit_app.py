"""NeuroSim Lab: run virtual experiments on real neurons from online databases.

Run locally with:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from neurosim import ephys, fitting
from neurosim import morphology as mo
from neurosim import simulation as sim
from neurosim import viz
from neurosim.sources import AllenCellTypesClient, NeuroMorphoClient

MAX_SECTIONS = 5000  # keep single runs to a few seconds on a small server
ANY = "Any"
FIT_DIR = Path.home() / ".cache" / "neurosim" / "fits"
ALLEN = "Allen Cell Types (with recordings)"
NEUROMORPHO = "NeuroMorpho.org database"
UPLOAD = "Upload an SWC file"
BALL_STICK = "Ball-and-stick model"

st.set_page_config(page_title="NeuroSim Lab", page_icon="🧠", layout="wide")


# ---------------------------------------------------------------------------
# Cached data access and simulation
# ---------------------------------------------------------------------------
@st.cache_resource
def nm_client() -> NeuroMorphoClient:
    return NeuroMorphoClient()


@st.cache_resource
def allen_client() -> AllenCellTypesClient:
    return AllenCellTypesClient()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def field_values(name: str) -> list[str]:
    try:
        return sorted(nm_client().field_values(name), key=str.lower)
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner="Searching NeuroMorpho.org…")
def nm_search(filters: tuple, page: int, size: int):
    return nm_client().search(page=page, size=size, **dict(filters))


@st.cache_data(ttl=24 * 3600, show_spinner="Downloading reconstruction…")
def nm_fetch(name: str):
    c = nm_client()
    rec = c.get(name)
    return rec, c.fetch_swc(rec)


@st.cache_data(ttl=7 * 24 * 3600, show_spinner="Loading the Allen Cell Types catalogue…")
def allen_cells():
    return allen_client().cells(reconstructed=True)


@st.cache_data(ttl=7 * 24 * 3600, show_spinner="Downloading reconstruction from the Allen Institute…")
def allen_swc(specimen_id: int) -> str:
    return allen_client().fetch_swc(specimen_id)


@st.cache_data(ttl=7 * 24 * 3600, show_spinner="Loading sweep list…")
def allen_sweeps(specimen_id: int):
    return allen_client().sweeps(specimen_id)


@st.cache_data(max_entries=64, show_spinner="Streaming the recording from the Allen Institute…")
def allen_trace(specimen_id: int, sweep_number: int):
    c = allen_client()
    sweep = next(s for s in allen_sweeps(specimen_id) if s.number == sweep_number)
    return c.trace(specimen_id, sweep)


@st.cache_resource(max_entries=8, show_spinner="Analysing morphology with NeuroM…")
def analyse(swc_text: str) -> dict:
    m = mo.load(swc_text)
    radii, counts = mo.sholl(m, step_size=10)
    return {
        "morph": m,
        "summary": mo.summary(m),
        "by_type": mo.per_neurite_type(m),
        "sholl": (radii, counts),
        "dist": mo.distributions(m),
        "geometry": mo.geometry(m),
    }


def _build_cell(model: dict, bio: dict):
    b = sim.Biophysics(**bio)
    if model["kind"] == "ball_and_stick":
        return sim.BallAndStick(**model["params"], bio=b)
    return sim.MorphologyCell(analyse(model["swc"])["morph"], b)


@st.cache_data(max_entries=64, show_spinner="Simulating with NEURON…")
def run_clamp(model: dict, bio: dict, stim: dict, record_dt: float | None):
    cell = _build_cell(model, bio)
    res = cell.run(sim.CurrentClamp(**stim), record_sections_dt=record_dt)
    info = {"compartments": getattr(cell, "n_compartments", None)}
    del cell
    return res, info


@st.cache_data(max_entries=32, show_spinner="Running f-I protocol…")
def run_fi(model: dict, bio: dict, stim: dict, amps: tuple):
    cell = _build_cell(model, bio)
    out = sim.fi_curve(cell, amps, sim.CurrentClamp(**stim))
    del cell
    return out


@st.cache_data(max_entries=32, show_spinner="Searching for rheobase…")
def run_rheobase(model: dict, bio: dict, stim: dict):
    cell = _build_cell(model, bio)
    out = sim.rheobase(cell, hi=5.0, tol=0.005, stim=sim.CurrentClamp(**stim))
    del cell
    return out


def load_fit(specimen_id: int) -> fitting.FitResult | None:
    path = FIT_DIR / f"{specimen_id}.json"
    if path.exists():
        try:
            return fitting.from_dict(json.loads(path.read_text()))
        except Exception:
            return None
    return None


def save_fit(specimen_id: int, res: fitting.FitResult) -> None:
    FIT_DIR.mkdir(parents=True, exist_ok=True)
    (FIT_DIR / f"{specimen_id}.json").write_text(json.dumps(fitting.to_dict(res)))


# ---------------------------------------------------------------------------
# Sidebar: source and stimulus (the membrane model comes later, once we know
# whether a fitted model exists for the selected cell)
# ---------------------------------------------------------------------------
st.sidebar.title("🧠 NeuroSim Lab")
source = st.sidebar.radio(
    "Neuron",
    [ALLEN, NEUROMORPHO, UPLOAD, BALL_STICK],
    help="Allen cells come with real patch-clamp recordings to compare against. NeuroMorpho.org has "
         "300k+ reconstructions from labs worldwide.",
)

st.sidebar.subheader("Stimulus: current clamp")
amp = st.sidebar.slider("Amplitude (nA)", -0.5, 3.0, 0.5 if source != ALLEN else 0.2, 0.01)
c1, c2 = st.sidebar.columns(2)
delay = c1.number_input("Delay (ms)", 0.0, 500.0, 20.0, 5.0)
dur = c2.number_input("Duration (ms)", 1.0, 1000.0, 100.0, 10.0)
tstop = st.sidebar.number_input("Simulation length (ms)", 10.0, 1500.0, max(150.0, delay + dur + 30), 10.0)
stim = asdict(sim.CurrentClamp(amp=amp, delay=delay, dur=dur, tstop=tstop))


# ---------------------------------------------------------------------------
# Choose the neuron
# ---------------------------------------------------------------------------
def neuromorpho_picker():
    """Search UI. Returns (record, swc_text) for the selected neuron, or None."""
    st.subheader("Find a neuron")
    cols = st.columns(4)
    wanted = {"species": "mouse", "brain_region": "neocortex", "cell_type": "pyramidal", "archive": ANY}
    labels = {"species": "Species", "brain_region": "Brain region", "cell_type": "Cell type", "archive": "Lab / archive"}
    filters = {}
    for col, (field, default) in zip(cols, wanted.items()):
        options = [ANY] + field_values(field)
        index = options.index(default) if default in options else 0
        choice = col.selectbox(labels[field], options, index=index, key=f"f_{field}")
        if choice != ANY:
            filters[field] = choice

    c1, c2, _ = st.columns([2, 1, 3])
    name_query = c1.text_input("…or open a neuron by name", placeholder="e.g. n419")
    page_size = 25

    if name_query.strip():
        try:
            return nm_fetch(name_query.strip())
        except Exception as e:
            st.error(f"Could not load neuron “{name_query}”: {e}")
            return None

    try:
        first = nm_search(tuple(sorted(filters.items())), 0, page_size)
    except Exception as e:
        st.error(f"NeuroMorpho.org search failed: {e}")
        return None
    if first.total == 0:
        st.warning("No neurons match these filters.")
        return None
    page = c2.number_input(f"Page (of {first.total_pages})", 1, max(first.total_pages, 1), 1) - 1
    res = first if page == 0 else nm_search(tuple(sorted(filters.items())), page, page_size)

    st.caption(f"**{res.total:,}** matching neurons. Click a row to load it.")
    df = pd.DataFrame([{
        "Name": n.name,
        "Species": n.species,
        "Brain region": ", ".join(n.brain_region),
        "Cell type": ", ".join(n.cell_type),
        "Lab": n.archive,
        "Reconstructed": n.domain,
    } for n in res.neurons])
    event = st.dataframe(
        df, hide_index=True, width="stretch", height=260,
        on_select="rerun", selection_mode="single-row", key=f"results_{hash(tuple(filters.items()))}_{page}",
    )
    row = event.selection.rows[0] if event.selection.rows else 0
    try:
        return nm_fetch(res.neurons[row].name)
    except Exception as e:
        st.error(f"Could not download {res.neurons[row].name}: {e}")
        return None


def allen_picker():
    """Allen Cell Types search. Returns the selected AllenCell, or None."""
    st.subheader("Find a recorded neuron")
    st.caption("Every cell here was recorded with patch-clamp electrophysiology and reconstructed by the "
               "[Allen Institute](https://celltypes.brain-map.org), so simulations can be checked against reality.")
    try:
        cells = allen_cells()
    except Exception as e:
        st.error(f"Could not reach the Allen Cell Types API: {e}")
        return None

    c1, c2, c3, c4 = st.columns(4)
    species = c1.selectbox("Species", [ANY, "mouse", "human"], index=1)
    dtype = c2.selectbox("Dendrite type", [ANY, "spiny", "aspiny", "sparsely spiny"],
                         help="Spiny cells are mostly excitatory (e.g. pyramidal); aspiny cells are mostly "
                              "inhibitory interneurons.")
    pool = [c for c in cells if (species == ANY or c.species == species) and (dtype == ANY or c.dendrite_type == dtype)]
    regions = sorted({c.region for c in pool})
    region = c3.selectbox("Brain region", [ANY] + regions)
    recon = c4.selectbox("Reconstruction", [ANY, "full", "dendrite-only"],
                         help="Full reconstructions include the axon.")
    pool = [c for c in pool if (region == ANY or c.region == region) and (recon == ANY or c.reconstruction == recon)]
    if not pool:
        st.warning("No cells match these filters.")
        return None

    fitted = {int(p.stem) for p in FIT_DIR.glob("*.json")} if FIT_DIR.exists() else set()
    st.caption(f"**{len(pool):,}** matching cells. Click a row to load it. ✓ = a fitted model is already saved.")
    df = pd.DataFrame([{
        "Specimen": c.specimen_id,
        "Fitted": "✓" if c.specimen_id in fitted else "",
        "Species": c.species,
        "Region": c.region_name or c.region,
        "Layer": c.layer,
        "Dendrites": c.dendrite_type,
        "Cre line": c.line,
        "Rest (mV)": c.features.get("vrest"),
        "Rin (MΩ)": c.features.get("ri"),
        "τ (ms)": c.features.get("tau"),
        "Rheobase (pA)": c.features.get("threshold_i_long_square"),
        "Max rate (Hz)": c.features.get("avg_firing_rate"),
    } for c in pool])
    event = st.dataframe(
        df, hide_index=True, width="stretch", height=260, on_select="rerun", selection_mode="single-row",
        key=f"allen_{species}_{dtype}_{region}_{recon}",
        column_config={
            "Specimen": st.column_config.NumberColumn(format="%d"),
            "Rest (mV)": st.column_config.NumberColumn(format="%.1f"),
            "Rin (MΩ)": st.column_config.NumberColumn(format="%.0f"),
            "τ (ms)": st.column_config.NumberColumn(format="%.1f"),
            "Rheobase (pA)": st.column_config.NumberColumn(format="%.0f"),
            "Max rate (Hz)": st.column_config.NumberColumn(format="%.0f"),
        },
    )
    row = event.selection.rows[0] if event.selection.rows else 0
    return pool[row]


nm_record, allen_cell, swc_text, model = None, None, None, None
st.title("Virtual experiments on real neurons")

if source == ALLEN:
    allen_cell = allen_picker()
    if allen_cell:
        try:
            swc_text = allen_swc(allen_cell.specimen_id)
        except Exception as e:
            st.error(f"Could not download the reconstruction: {e}")
elif source == NEUROMORPHO:
    picked = neuromorpho_picker()
    if picked:
        nm_record, swc_text = picked
elif source == UPLOAD:
    up = st.file_uploader("SWC reconstruction", type=["swc"])
    if up:
        swc_text = up.getvalue().decode("utf-8", errors="replace").replace("\r\n", "\n")
    else:
        st.info("Upload an `.swc` morphology to analyse and simulate it.")
else:
    st.subheader("Ball-and-stick model")
    st.caption("The classic teaching model from the EPFL *Simulation Neuroscience* course: "
               "a Hodgkin-Huxley soma attached to a single passive dendrite.")
    c1, c2, c3 = st.columns(3)
    model = {"kind": "ball_and_stick", "params": {
        "soma_diam": c1.slider("Soma diameter (µm)", 5.0, 60.0, 20.0, 1.0),
        "dend_length": c2.slider("Dendrite length (µm)", 10.0, 2000.0, 200.0, 10.0),
        "dend_diam": c3.slider("Dendrite diameter (µm)", 0.2, 5.0, 1.0, 0.1),
    }}

analysis = None
if swc_text:
    try:
        analysis = analyse(swc_text)
    except Exception as e:
        st.error(f"NeuroM could not read this reconstruction: {e}")
        st.stop()
    model = {"kind": "morphology", "swc": swc_text}

# ---------------------------------------------------------------------------
# Sidebar: membrane model (fitted, if available, or hand-tuned)
# ---------------------------------------------------------------------------
fit = load_fit(allen_cell.specimen_id) if allen_cell else None
use_fit = False
if fit:
    use_fit = st.sidebar.toggle("Use the model fitted to this cell", True,
                                help="Parameters fitted to this cell's recorded resting potential, input "
                                     "resistance, time constant and f-I curve.")

defaults = sim.Biophysics()
if use_fit:
    bio = asdict(fit.bio)
    with st.sidebar.expander("Fitted membrane model"):
        b = fit.bio
        st.markdown(
            f"- Leak g_pas: **{b.g_pas:.2e}** S/cm², E_leak **{b.e_pas:.1f}** mV\n"
            f"- Capacitance cm: **{b.cm:.2f}** µF/cm²\n"
            f"- ḡNa **{b.gnabar:.3f}**, ḡK **{b.gkbar:.3f}** S/cm² (× {b.soma_hh_scale:g} soma, "
            f"× {b.ais_hh_scale:g} AIS)\n"
            f"- Kinetics temperature factor: **{b.celsius:g} °C**"
        )
else:
    with st.sidebar.expander("Membrane model"):
        st.caption("Hodgkin-Huxley soma/axon with passive dendrites. Generic parameters, not fitted to the cell.")
        gnabar = st.slider("Na conductance ḡNa (S/cm²)", 0.0, 0.5, defaults.gnabar, 0.01)
        gkbar = st.slider("K conductance ḡK (S/cm²)", 0.0, 0.2, defaults.gkbar, 0.002)
        g_pas = st.number_input("Dendritic leak g_pas (S/cm²)", 1e-6, 1e-3, defaults.g_pas, format="%.1e")
        Ra = st.slider("Axial resistivity Ra (Ω·cm)", 30.0, 400.0, defaults.Ra, 10.0)
        celsius = st.slider("Temperature (°C)", -10.0, 37.0, defaults.celsius, 0.1)
        dend_scale = st.slider("Active dendrites (HH density × )", 0.0, 1.0, 0.0, 0.05,
                               help="0 = passive dendrites. >0 inserts HH channels at this fraction of somatic density.")
        if source != BALL_STICK:
            soma_scale = st.slider("Somatic channel density ×", 0.0, 10.0, defaults.soma_hh_scale, 0.5)
            ais_scale = st.slider("Axon initial segment density ×", 0.0, 50.0, defaults.ais_hh_scale, 1.0,
                                  help="Spikes start in the AIS. Set 0 to remove it.")
        else:
            soma_scale, ais_scale = 1.0, 0.0
    bio = asdict(sim.Biophysics(
        gnabar=gnabar, gkbar=gkbar, g_pas=g_pas, Ra=Ra, celsius=celsius, dendrite_hh_scale=dend_scale,
        soma_hh_scale=soma_scale, ais_hh_scale=ais_scale,
    ))
st.sidebar.caption(
    "Data: [Allen Cell Types](https://celltypes.brain-map.org) · [NeuroMorpho.org](https://neuromorpho.org) · "
    "Analysis: [NeuroM](https://github.com/openbraininstitute/NeuroM), "
    "[eFEL](https://github.com/openbraininstitute/eFEL) · Simulation: [NEURON](https://nrn.readthedocs.io)"
)

if model is None:
    st.stop()

n_sections = int(analysis["summary"]["number_of_sections"]) if analysis else 1
can_simulate = n_sections <= MAX_SECTIONS

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
if nm_record:
    st.markdown(
        f"### {nm_record.name}  \n"
        f"{nm_record.species} · {', '.join(nm_record.brain_region) or 'unknown region'} · "
        f"{', '.join(nm_record.cell_type) or 'unknown type'} · lab: {nm_record.archive} · "
        f"[NeuroMorpho page]({nm_record.page_url})"
    )
if allen_cell:
    st.markdown(
        f"### Allen cell {allen_cell.specimen_id}  \n"
        f"{allen_cell.species} · {allen_cell.region_name or allen_cell.region}"
        f"{'' if 'layer' in allen_cell.region_name.lower() else f', layer {allen_cell.layer}'} · "
        f"{allen_cell.dendrite_type} · {allen_cell.line or 'no Cre line'} · "
        f"[Allen Cell Types page]({allen_cell.page_url})"
    )

tab_names = (["Compare with recording"] if allen_cell else []) + (["Morphology"] if analysis else []) \
    + ["Current clamp", "f-I curve", "About"]
tabs = dict(zip(tab_names, st.tabs(tab_names)))

# ---------------------------------------------------------------------------
# Compare with recording (Allen cells only)
# ---------------------------------------------------------------------------
if allen_cell:
    with tabs["Compare with recording"]:
        sid = allen_cell.specimen_id
        try:
            sweeps = allen_sweeps(sid)
        except Exception as e:
            st.error(f"Could not load the sweep list: {e}")
            sweeps = []
        long_sq = [s for s in sweeps if s.stimulus == "Long Square" and s.start_s is not None]
        targets = fitting.targets_from_allen(allen_cell, sweeps) if sweeps else None

        # --- fitting -------------------------------------------------------
        st.markdown("#### 1 · Fit the model to this cell")
        if fit is None:
            st.markdown(
                "The simulation below uses generic textbook parameters, so it won't look like the recording yet. "
                "Fitting tunes the leak, capacitance, Na/K densities and channel kinetics so the model "
                "matches this cell's resting potential, input resistance, time constant, rheobase and f-I curve."
            )
            if not can_simulate:
                st.warning("This reconstruction is too large to fit here.")
            elif targets is None or targets.rheobase is None:
                st.warning("This cell has no usable long-square recordings to fit.")
            elif st.button("Fit model to this cell (≈1–2 min)", type="primary"):
                with st.status("Fitting…", expanded=False) as status:
                    res = fitting.fit(analysis["morph"], targets, progress=lambda m: status.update(label=m))
                    save_fit(sid, res)
                    status.update(label="Fit complete", state="complete")
                st.rerun()
        else:
            t, a = fit.targets, fit.achieved
            rows = [
                ("Resting potential", "mV", t.vrest, a["vrest"]),
                ("Input resistance", "MΩ", t.ri, a["ri"]),
                ("Membrane time constant", "ms", t.tau, a["tau"]),
                ("Rheobase", "pA", (t.rheobase or np.nan) * 1000, a["rheobase"] * 1000),
            ]
            c1, c2 = st.columns([2, 3])
            with c1:
                st.dataframe(pd.DataFrame(rows, columns=["Property", "Unit", "Recorded", "Fitted model"]),
                             hide_index=True, width="stretch",
                             column_config={"Recorded": st.column_config.NumberColumn(format="%.1f"),
                                            "Fitted model": st.column_config.NumberColumn(format="%.1f")})
                for note in fit.notes:
                    st.caption(f"ℹ️ {note}")
                if not use_fit:
                    st.info("The fitted model is switched off in the sidebar.")
            with c2:
                st.caption("f-I curve: firing rate on 1 s current steps")
                st.plotly_chart(viz.fi_comparison_figure(t.fi_amps, t.fi_rates, fit.fi_rates_model),
                                width="stretch")

        # --- sweep comparison ----------------------------------------------
        st.markdown("#### 2 · Replay a recorded experiment")
        if not long_sq:
            st.info("No long-square current steps were recorded for this cell.")
        elif can_simulate:
            amps_pa = sorted({round(s.amplitude_pa) for s in long_sq})
            spiking = [a for a in amps_pa if any(s.num_spikes > 0 and round(s.amplitude_pa) == a for s in long_sq)]
            default = spiking[min(2, len(spiking) - 1)] if spiking else amps_pa[-1]
            chosen = st.select_slider("Injected current (pA)", amps_pa, value=default,
                                      help="Each value is a 1 s current step recorded from this cell.")
            sweep = next(s for s in long_sq if round(s.amplitude_pa) == chosen)
            step_ms = sweep.duration_s * 1000
            try:
                trace = allen_trace(sid, sweep.number)
            except Exception as e:
                st.error(f"Could not stream the recording: {e}")
                trace = None
            if trace is not None:
                pre = 100.0
                sim_stim = asdict(sim.CurrentClamp(amp=chosen / 1000, delay=pre, dur=step_ms,
                                                   tstop=pre + step_ms + 200, dt=fitting.FIT_DT))
                res, _ = run_clamp(model, bio, sim_stim, None)
                t_sim, v_sim = res.t - pre, res.v["soma"]
                st.plotly_chart(viz.comparison_figure(
                    trace.t, trace.v, t_sim, v_sim,
                    title=f"Sweep {sweep.number}: {chosen} pA for {step_ms:.0f} ms "
                          f"({'fitted' if use_fit else 'generic'} model)",
                ), width="stretch")
                # input resistance is only meaningful on hyperpolarising (non-spiking) steps
                amp_for_ri = chosen / 1000 if chosen < 0 else None
                f_rec = ephys.features(trace.t, trace.v, 0.0, step_ms, amp_for_ri)
                f_sim = ephys.features(t_sim, v_sim, 0.0, step_ms, amp_for_ri)
                table = pd.DataFrame(ephys.compare(f_rec, f_sim))
                st.caption("Electrophysiology features computed with eFEL on both traces")
                st.dataframe(table, hide_index=True, width="stretch", column_config={
                    c: st.column_config.NumberColumn(format="%.2f") for c in ("Recorded", "Simulated", "Difference")
                })
        else:
            st.warning("This reconstruction is too large to simulate here.")

# ---------------------------------------------------------------------------
# Morphology
# ---------------------------------------------------------------------------
if analysis:
    with tabs["Morphology"]:
        s = analysis["summary"]
        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(viz.morphology_figure(analysis["geometry"]), width="stretch")
        with right:
            m1, m2 = st.columns(2)
            m1.metric("Total length", f"{s['total_length']:,.0f} µm")
            m2.metric("Branch points", f"{s['number_of_bifurcations']:.0f}")
            m1.metric("Sections", f"{s['number_of_sections']:.0f}")
            m2.metric("Neurites", f"{s['number_of_neurites']:.0f}")
            m1.metric("Soma radius", f"{s['soma_radius']:.1f} µm")
            m2.metric("Max reach", f"{s['max_radial_distance']:,.0f} µm")
            if analysis["by_type"]:
                df = pd.DataFrame(analysis["by_type"]).T
                df.index = [viz.NEURITE_LABELS.get(i, i) for i in df.index]
                df.columns = ["Neurites", "Sections", "Branches", "Length (µm)", "Reach (µm)"]
                st.dataframe(df.style.format("{:,.0f}"), width="stretch")
            if nm_record and nm_record.png_url:
                st.image(nm_record.png_url, caption="NeuroMorpho.org preview", width="stretch")
            if nm_record and nm_record.reference_doi:
                st.caption("Reference: " + " · ".join(f"[{d}](https://doi.org/{d})" for d in nm_record.reference_doi))

        st.markdown("#### Morphometrics")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption("Sholl analysis: branches crossing spheres around the soma")
            st.plotly_chart(viz.sholl_figure(*analysis["sholl"]), width="stretch")
        with c2:
            st.caption("Section lengths")
            st.plotly_chart(viz.histogram_figure(analysis["dist"]["section_lengths"], "length (µm)"),
                            width="stretch")
        with c3:
            st.caption("Bifurcation angles")
            st.plotly_chart(viz.histogram_figure(np.degrees(analysis["dist"]["remote_bifurcation_angles"]),
                                                 "angle (°)"), width="stretch")

# ---------------------------------------------------------------------------
# Current clamp
# ---------------------------------------------------------------------------
with tabs["Current clamp"]:
    if not can_simulate:
        st.warning(f"This reconstruction has {n_sections:,} sections; simulations here are limited to "
                   f"{MAX_SECTIONS:,}. Try a smaller neuron.")
    else:
        st.caption(f"Injecting **{amp:.2f} nA** into the soma for {dur:.0f} ms. "
                   "Change the stimulus and membrane model in the sidebar.")
        show_3d = analysis is not None and st.toggle("Show voltage spreading through the neuron (3D)", True)
        record_dt = max(stim["tstop"] / 200, 0.25) if show_3d else None
        try:
            res, info = run_clamp(model, bio, stim, record_dt)
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            st.stop()
        v = res.v["soma"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Spikes", len(res.spike_times))
        m2.metric("Firing rate", f"{res.firing_rate:.0f} Hz")
        m3.metric("Peak soma V", f"{v.max():.1f} mV")
        m4.metric("Resting V", f"{v[res.t < delay].mean() if delay > 0 else v[0]:.1f} mV")
        st.plotly_chart(viz.traces_figure(res), width="stretch")
        if show_3d:
            vmin = min(-80.0, float(bio["v_init"]) - 5)
            st.caption("Press ▶ to watch the membrane potential propagate (colour = mV).")
            st.plotly_chart(viz.voltage_figure(analysis["geometry"], res, vmin=vmin), width="stretch")
        if info.get("compartments"):
            st.caption(f"Model: {info['compartments']:,} compartments, dt = {res.stim.dt} ms.")

# ---------------------------------------------------------------------------
# f-I curve
# ---------------------------------------------------------------------------
with tabs["f-I curve"]:
    st.caption("Repeat the current step at increasing amplitudes and count spikes: the cell's input-output function.")
    if not can_simulate:
        st.warning("This neuron is too large to simulate here.")
    else:
        c1, c2, c3 = st.columns(3)
        lo, hi = c1.slider("Current range (nA)", 0.0, 3.0, (0.0, 0.5 if source == ALLEN else 1.5), 0.05)
        steps = c2.slider("Steps", 3, 25, 10)
        amps = tuple(np.round(np.linspace(lo, hi, steps), 4))
        if st.button("Run f-I curve", type="primary"):
            st.session_state["fi_request"] = (model, bio, stim, amps)
        if c3.button("Find rheobase"):
            rb = run_rheobase(model, bio, stim)
            c3.metric("Rheobase", "no spikes ≤ 5 nA" if rb is None else f"{rb:.3f} nA")
        req = st.session_state.get("fi_request")
        if req and req == (model, bio, stim, amps):
            x, y = run_fi(*req)
            st.plotly_chart(viz.fi_figure(x, y), width="stretch")
        elif req:
            st.info("Parameters changed. Press **Run f-I curve** to update.")

# ---------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------
with tabs["About"]:
    st.markdown("""
**How it works**

1. **Data**: neurons come live from the [Allen Cell Types Database](https://celltypes.brain-map.org)
   (patch-clamp recordings + reconstructions) or [NeuroMorpho.org](https://neuromorpho.org)
   (the largest public archive of reconstructed neurons). Allen recordings are streamed from the
   NWB files sweep by sweep, so only the requested trace is downloaded.
2. **Morphology**: files are parsed and measured with [NeuroM](https://github.com/openbraininstitute/NeuroM).
   Common soma problems in database files are repaired automatically.
3. **Simulation**: each branch becomes a NEURON section with its real 3D geometry, split into compartments
   by the d-lambda rule. The soma and axon carry Hodgkin-Huxley Na/K channels, dendrites are passive by
   default, and an axon initial segment with a high channel density is where spikes start.
4. **Fitting** (Allen cells): a uniform leak and capacitance are fitted to the recorded resting potential,
   input resistance and time constant; Na/K densities and channel kinetic speed are fitted to the recorded
   rheobase and f-I curve. Recorded and simulated traces are compared with
   [eFEL](https://github.com/openbraininstitute/eFEL) features.

**Caveats**: the channels are squid-axon Hodgkin-Huxley channels, not the mix of channels real cortical
neurons express. Fitted models match passive properties, rheobase and firing rates well, but spike shape only
qualitatively. Some reconstructions (e.g. with a placeholder 1 µm soma or missing branches) give
unphysiological input resistances or capacitances.

Source code: [github.com/cdjcodes/neuroscience](https://github.com/cdjcodes/neuroscience)
""")
