"""NeuroSim Lab: run virtual experiments on real neurons from NeuroMorpho.org.

Run locally with:  streamlit run streamlit_app.py
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import streamlit as st

from neurosim import morphology as mo
from neurosim import simulation as sim
from neurosim import viz
from neurosim.sources import NeuroMorphoClient

MAX_SECTIONS = 5000  # keep single runs to a few seconds on a small server
ANY = "Any"

st.set_page_config(page_title="NeuroSim Lab", page_icon="🧠", layout="wide")


# ---------------------------------------------------------------------------
# Cached data access and simulation
# ---------------------------------------------------------------------------
@st.cache_resource
def client() -> NeuroMorphoClient:
    return NeuroMorphoClient()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def field_values(name: str) -> list[str]:
    try:
        return sorted(client().field_values(name), key=str.lower)
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner="Searching NeuroMorpho.org…")
def search(filters: tuple, page: int, size: int):
    return client().search(page=page, size=size, **dict(filters))


@st.cache_data(ttl=24 * 3600, show_spinner="Downloading reconstruction…")
def fetch(name: str):
    c = client()
    rec = c.get(name)
    return rec, c.fetch_swc(rec)


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


# ---------------------------------------------------------------------------
# Sidebar: neuron source and experiment parameters
# ---------------------------------------------------------------------------
st.sidebar.title("🧠 NeuroSim Lab")
source = st.sidebar.radio(
    "Neuron",
    ["NeuroMorpho.org database", "Upload an SWC file", "Ball-and-stick model"],
    help="Use a real reconstructed neuron, your own reconstruction, or the classic teaching model.",
)

st.sidebar.subheader("Stimulus: current clamp")
amp = st.sidebar.slider("Amplitude (nA)", -0.5, 3.0, 0.5, 0.01)
c1, c2 = st.sidebar.columns(2)
delay = c1.number_input("Delay (ms)", 0.0, 500.0, 20.0, 5.0)
dur = c2.number_input("Duration (ms)", 1.0, 1000.0, 100.0, 10.0)
tstop = st.sidebar.number_input("Simulation length (ms)", 10.0, 1500.0, max(150.0, delay + dur + 30), 10.0)
stim = asdict(sim.CurrentClamp(amp=amp, delay=delay, dur=dur, tstop=tstop))

defaults = sim.Biophysics()
with st.sidebar.expander("Membrane model"):
    st.caption("Hodgkin-Huxley soma/axon with passive dendrites. Generic parameters, not fitted to the cell.")
    gnabar = st.slider("Na conductance ḡNa (S/cm²)", 0.0, 0.5, defaults.gnabar, 0.01)
    gkbar = st.slider("K conductance ḡK (S/cm²)", 0.0, 0.2, defaults.gkbar, 0.002)
    g_pas = st.number_input("Dendritic leak g_pas (S/cm²)", 1e-6, 1e-3, defaults.g_pas, format="%.1e")
    Ra = st.slider("Axial resistivity Ra (Ω·cm)", 30.0, 400.0, defaults.Ra, 10.0)
    celsius = st.slider("Temperature (°C)", 6.3, 37.0, defaults.celsius, 0.1)
    dend_scale = st.slider("Active dendrites (HH density × )", 0.0, 1.0, 0.0, 0.05,
                           help="0 = passive dendrites. >0 inserts HH channels at this fraction of somatic density.")
    if source != "Ball-and-stick model":
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
    "Data: [NeuroMorpho.org](https://neuromorpho.org) · Analysis: "
    "[NeuroM](https://github.com/openbraininstitute/NeuroM) · Simulation: [NEURON](https://nrn.readthedocs.io)"
)


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
            return fetch(name_query.strip())
        except Exception as e:
            st.error(f"Could not load neuron “{name_query}”: {e}")
            return None

    try:
        first = search(tuple(sorted(filters.items())), 0, page_size)
    except Exception as e:
        st.error(f"NeuroMorpho.org search failed: {e}")
        return None
    if first.total == 0:
        st.warning("No neurons match these filters.")
        return None
    page = c2.number_input(f"Page (of {first.total_pages})", 1, max(first.total_pages, 1), 1) - 1
    res = first if page == 0 else search(tuple(sorted(filters.items())), page, page_size)

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
        return fetch(res.neurons[row].name)
    except Exception as e:
        st.error(f"Could not download {res.neurons[row].name}: {e}")
        return None


record, swc_text, model = None, None, None
st.title("Virtual experiments on real neurons")

if source == "NeuroMorpho.org database":
    picked = neuromorpho_picker()
    if picked:
        record, swc_text = picked
elif source == "Upload an SWC file":
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

if model is None:
    st.stop()

n_sections = int(analysis["summary"]["number_of_sections"]) if analysis else 1
can_simulate = n_sections <= MAX_SECTIONS

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if record:
    st.markdown(
        f"### {record.name}  \n"
        f"{record.species} · {', '.join(record.brain_region) or 'unknown region'} · "
        f"{', '.join(record.cell_type) or 'unknown type'} · lab: {record.archive} · "
        f"[NeuroMorpho page]({record.page_url})"
    )

tab_names = (["Morphology"] if analysis else []) + ["Current clamp", "f-I curve", "About"]
tabs = dict(zip(tab_names, st.tabs(tab_names)))

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
                df.columns = ["Neurites", "Sections", "Branch points", "Length (µm)", "Max reach (µm)"]
                st.dataframe(df.style.format("{:,.0f}"), width="stretch")
            if record and record.png_url:
                st.image(record.png_url, caption="NeuroMorpho.org preview", width="stretch")
            if record and record.reference_doi:
                st.caption("Reference: " + " · ".join(f"[{d}](https://doi.org/{d})" for d in record.reference_doi))

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
            st.caption("Press ▶ to watch the membrane potential propagate (colour = mV).")
            st.plotly_chart(viz.voltage_figure(analysis["geometry"], res), width="stretch")
        if info.get("compartments"):
            st.caption(f"Model: {info['compartments']:,} compartments, dt = {res.stim.dt} ms.")

with tabs["f-I curve"]:
    st.caption("Repeat the current step at increasing amplitudes and count spikes: the cell's input-output function.")
    if not can_simulate:
        st.warning("This neuron is too large to simulate here.")
    else:
        c1, c2, c3 = st.columns(3)
        lo, hi = c1.slider("Current range (nA)", 0.0, 3.0, (0.0, 1.5), 0.05)
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

with tabs["About"]:
    st.markdown("""
**How it works**

1. **Data**: neuron reconstructions are searched and downloaded live from
   [NeuroMorpho.org](https://neuromorpho.org), the largest public archive of digitally reconstructed neurons.
2. **Morphology**: files are parsed and measured with [NeuroM](https://github.com/openbraininstitute/NeuroM).
   Common soma problems in database files are repaired automatically.
3. **Simulation**: each branch becomes a NEURON section with its real 3D geometry, split into compartments
   by the d-lambda rule. The soma and axon carry Hodgkin-Huxley Na/K channels, dendrites are passive by
   default, and an axon initial segment with a high channel density is where spikes start.

**Caveats**: the membrane model is generic, not fitted to the recorded behaviour of each cell,
so treat the results as *what this shape does with textbook channels*. Some reconstructions
(e.g. with a placeholder 1 µm soma) give unphysiological input resistances.

Source code: [github.com/cdjcodes/neuroscience](https://github.com/cdjcodes/neuroscience)
""")
