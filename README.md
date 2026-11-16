![](banner.png)
<!-- PROJECT SHIELDS -->
![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)
[![tests](https://github.com/cdjcodes/neuroscience/actions/workflows/tests.yml/badge.svg)](https://github.com/cdjcodes/neuroscience/actions/workflows/tests.yml)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://neurosimlabcdj.streamlit.app/)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/cdjcodes/neuroscience)
![GitHub contributors](https://img.shields.io/github/contributors-anon/cdjcodes/neuroscience)

Personal repository for all Neuroscience related projects. 

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Contents</summary>
  <ol>
    <li><a href="#neurosim">neurosim: experiments on real neurons</a></li>
    <li>
      <a href="#sim-neuro">Simulation Neuroscience</a>
      <ul>
        <li><a href="#reading-neuro-data">Accessing and Reading Neuroscience Data</a></li>
      </ul>
      <ul>
        <li><a href="#neuron-morph">Neuron Morphologies</a></li>
      </ul>
      <ul>
        <li><a href="#ball-and-stick">Ball and Stick Model</a></li>
      </ul>
      <ul>
        <li><a href="#neuron-sim">Neuron Simulations</a></li>
      </ul>
    </li>
  </ol>
</details>


<!-- neurosim -->
<a name="neurosim"></a>
# neurosim: experiments on real neurons

**🧠 Try it live: [neurosimlabcdj.streamlit.app](https://neurosimlabcdj.streamlit.app/)**

`neurosim` is a Python package that turns the course notebooks below into reusable code for running virtual experiments on **real neurons from online databases**, and checking them against **real recordings**:

| Module | What it does | Built on |
| --- | --- | --- |
| `neurosim.sources` | Search [NeuroMorpho.org](https://neuromorpho.org) (300k+ reconstructed neurons) and the [Allen Cell Types Database](https://celltypes.brain-map.org) (667 neurons with both a reconstruction and patch-clamp recordings). Allen recordings are streamed sweep by sweep from the NWB files with HTTP range requests, so only the requested trace is downloaded. Everything is cached locally | NeuroMorpho & Allen REST APIs, h5py |
| `neurosim.morphology` | Load reconstructions (repairing the common soma problems in database files), compute morphometrics, Sholl analysis, distributions, 3D geometry for plotting | [NeuroM](https://github.com/openbraininstitute/NeuroM) |
| `neurosim.simulation` | Current-clamp experiments, f-I curves and rheobase, on the classic ball-and-stick model, a Hodgkin-Huxley multi-compartment model built from any downloaded neuron, or the Allen Institute's published biophysical model of a cell | [NEURON](https://nrn.readthedocs.io) |
| `neurosim.channels` | Cortical ion-channel models (Na, K, Ca, Ih) from the Allen Institute's perisomatic models, pinned and reviewed in `neurosim/mechanisms/`, compiled on first use with `nrnivmodl` | NEURON + C compiler |
| `neurosim.ephys` | Electrophysiology features (spike count, rate, latency, threshold, spike shape, adaptation, input resistance) computed identically on recorded and simulated traces | [eFEL](https://github.com/openbraininstitute/eFEL) |
| `neurosim.fitting` | Fit a model to a recorded cell: resting potential, input resistance, membrane time constant, rheobase and f-I curve | NEURON |

### Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[app,dev]"
pytest                      # add -m "not network" to skip tests that hit NeuroMorpho.org
```

```python
from neurosim.sources import NeuroMorphoClient
from neurosim import morphology, simulation as sim

client = NeuroMorphoClient()
hits = client.search(species="mouse", brain_region="neocortex", cell_type="pyramidal", size=5)
neuron = hits.neurons[0]

morph = morphology.load(client.fetch_swc(neuron))
print(morphology.summary(morph))              # total length, branch points, ...

cell = sim.MorphologyCell(morph)
result = cell.run(sim.CurrentClamp(amp=0.3, delay=20, dur=100, tstop=150))
print(result.firing_rate, "Hz")               # result.t, result.v["soma"], ...
print(sim.fi_curve(cell, [0.1, 0.2, 0.4, 0.8]))
```

Compare with a real recording from the Allen Cell Types Database:

```python
from neurosim.sources import AllenCellTypesClient
from neurosim import ephys, fitting

allen = AllenCellTypesClient()
cell = allen.get(502359001)                       # mouse V1 layer 2/3 pyramidal cell
sweeps = allen.sweeps(cell.specimen_id)
morph = morphology.load(allen.fetch_swc(cell))

fit = fitting.fit(morph, fitting.targets_from_allen(cell, sweeps))   # ~1 min
print(fit.achieved, fit.targets.vrest, fit.targets.ri, fit.targets.tau)

sweep = next(s for s in sweeps if s.stimulus == "Long Square" and s.num_spikes > 5)
recorded = allen.trace(cell, sweep)              # streams ~3 MB instead of the whole NWB file
model = sim.MorphologyCell(morph, fit.bio).run(
    sim.CurrentClamp(amp=sweep.amplitude_pa / 1000, delay=100, dur=1000, tstop=1300, dt=0.05))
print(ephys.compare(
    ephys.features(recorded.t, recorded.v, 0, 1000),
    ephys.features(model.t - 100, model.v["soma"], 0, 1000),
))
```

### Web app: NeuroSim Lab

```bash
streamlit run streamlit_app.py
```

The app opens at http://localhost:8501 and lets you:

- **Compare with recording** (Allen cells): replay any recorded current step and compare the recording with every available model: the Allen Institute's published biophysical model (cortical channels; 236 cells) and a Hodgkin-Huxley model you fit in the app in about a minute. The tab shows the voltage traces, a close-up of the first action potential, an eFEL feature table and recorded vs. model f-I curves.
- **Find a neuron**: browse Allen cells by species, dendrite type and region, or filter NeuroMorpho.org by species, brain region, cell type and lab (or type a neuron name), then click a row to load it. You can also upload your own `.swc` file or use the course's ball-and-stick model.
- **Morphology**: an interactive 3D view, key morphometrics, Sholl analysis and branch statistics.
- **Current clamp**: inject a current step and see the voltage at the soma, the farthest dendrite and the axon, plus an animated 3D view of the voltage spreading through the cell.
- **f-I curve**: firing rate against injected current, and the rheobase (the smallest current that makes the cell spike).

Stimulus and membrane parameters (channel densities, leak, axial resistance, temperature, active dendrites, AIS) are in the sidebar. Results are cached, so revisiting a setting is instant.

### Deploying the app

The app runs on [Streamlit Community Cloud](https://streamlit.io/cloud) (free) and redeploys automatically on every push to `main`. To deploy your own copy:

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub and click **Create app → Deploy a public app from GitHub**.
2. Repository `cdjcodes/neuroscience`, branch `main`, main file `streamlit_app.py`.
3. Under **Advanced settings**, choose Python **3.12**, then **Deploy**.

Dependencies come from `requirements.txt` and settings from `.streamlit/config.toml`. Downloads and fitted models are cached on the server's disk, which is wiped whenever the app restarts, so they are rebuilt on demand. GitHub Actions runs the test suite on every push (`.github/workflows/tests.yml`).

### Model notes

**Allen biophysical models** (`simulation.AllenPerisomaticCell`): for 236 reconstructed cells, the Allen Institute published a model fitted to that cell's recordings. It has cortical channels in the soma (transient and persistent Na, Kv3.1, K_T, K_P, Im, SK, HVA/LVA Ca with Ca dynamics, Ih), passive dendrites and an axon stub, simulated at 34 °C. The app rebuilds it following Allen's recipe, using the 16 channel files pinned in `neurosim/mechanisms/allen_perisomatic` (sources, terms of use and checksums in its README). They are compiled once with `nrnivmodl`, which needs a C compiler (`packages.txt` installs one on Streamlit Cloud). Allen fitted these models to junction-potential-corrected data, so their voltages are shifted by the fit's junction potential (+14 mV) to compare with the raw recordings. On the sweeps they were fitted to, they reproduce spike count, latency and spike shape closely (e.g. 1.14 vs 1.08 ms half-width). Near threshold they can differ more.

**Hodgkin-Huxley models:** the multi-compartment model is generic, **not fitted to the individual cell**: Hodgkin-Huxley channels in the soma and axon, passive dendrites, and an axon initial segment (AIS) with a high Na/K density where spikes start (a synthetic AIS is added if the reconstruction has no axon). Without the AIS, the large passive dendritic tree loads the soma so heavily that squid HH channels fire one spike and then block. All parameters are in `simulation.Biophysics`.

**Fitting** (`neurosim.fitting`) works in two cheap stages:
1. **Passive:** a uniform leak is fitted by bisection to the recorded input resistance and resting potential. Dendritic capacitance is fitted to the time constant, while the soma and axon keep 1 µF/cm². The extra dendritic capacitance stands in for spines, which reconstructions omit.
2. **Active:** Na density is bisected so the model starts firing at the recorded rheobase. A small grid over K density and channel kinetic speed picks the best match to the recorded f-I curve. Squid HH channels jump straight to ~40 Hz at threshold, whereas cortical neurons can fire slowly, and slowing the kinetics fixes most of that.

Fitted models match passive properties and rheobase closely, and firing rates reasonably well. Spike shape (height, width, threshold) stays squid-like because the channel types are not fitted. Cells with incomplete reconstructions (common for slice recordings) need an implausibly high capacitance, and the app flags this.

### Roadmap

1. ✅ Core library: database access, morphometrics, simulations
2. ✅ Web app (Streamlit): search the database, 3D viewer, run experiments interactively
3. ✅ Real electrophysiology: compare simulations against Allen Cell Types recordings (eFEL features) and fit models to them
4. ✅ Deployed online: [neurosimlabcdj.streamlit.app](https://neurosimlabcdj.streamlit.app/)
5. ✅ Realistic spike shapes: the Allen Institute's biophysical models with cortical ion channels

<!-- Simulation Neuroscience -->
# Simulation Neuroscience

This project contains all of the exercises and derived projects based on the MOOC Simulation Neuroscience by EPFL in edX.org. This course was designed to learn how to 
digitally reconstruct a neuron and further expand to simulate a connection between two neurons. Neuronal simulation is important in order to better study the biological mechanisms of brain function, behaviour and disease. The course can be accessed [here](https://www.edx.org/course/simulation-neuroscience).

**Main References:**

Original exercises done under this project and other references used in the course can be accessed in the following links:
1. [Blue Brain MOOC](https://github.com/BlueBrain/MOOC-neurons-and-synapses-2017)
2. [Blue Brain Nexus](https://bluebrainnexus.io/)
3. [Blue Brain Nexus Sandbox](https://sandbox.bluebrainnexus.io/)
4. [Allen Cell Types Database](https://celltypes.brain-map.org/) 
5. [AllenSDK](https://allensdk.readthedocs.io/en/latest/)
6. [MouseLight project](https://www.janelia.org/project-team/mouselight)
7. [NeuroM](https://github.com/openbraininstitute/NeuroM) (now maintained by the Open Brain Institute)
8. [NEURON](https://www.neuron.yale.edu/neuron/docs)
9. [NMODL](https://github.com/BlueBrain/nmodl)
10. [Blue Brain Python Cell Model Management](https://github.com/BlueBrain/BluePyMM)
11. [Blue Brain Python Optimisation Library](https://github.com/BlueBrain/BluePyOpt)
12. [Blue Brain eFel](https://github.com/BlueBrain/eFEL)
13. [The Blue Brain Project](https://github.com/BlueBrain)

## Accessing and Reading Neuroscience Data
This section contains notebooks which show how to access existing open source neuronal data such as that from the Allen Cell Type Database. This also contains information on how to organize various neuronal data  and how to create and manipulate metadata with the MINDS (Minimal Information for Neuroscience Datasets) framework using Blue Brain Nexus (data integration system). Nexus has three parts:

- Nexus Fusion : user-interface built on top of Delta to access, manage and manipulate neuronal data
- Nexus Forge : python-based program used for building knowledge-graphs, manipulating and reusing ontologies, schema and modifying semantics of datasets
- Nexus Delta : storage & management of knowledge graphs


## Neuron Morphologies
This section contains exercises on how to access neuronal data from various sources, visualize, obtain features, analyze neuronal morphometrics and do statistical analysis on neuron morphologies.

## Ball and Stick Model
This section involves modeling of neurons including their electrical, synaptic and ion-channel characteristics. The baseline for this model is the ball-and-stick model whose basis is the Hodgkin-Huxley model of axonal dynamics. We build on this model to incorporate synaptic potential dynamics (Rall's Cable Model) and transmission (EPSPs and IPSPs) from a pre-synaptic neuron to post-synaptic neuron. 
## Neuron Simulations
This section involves exploring how to leverage experimental data to constrain the neuronal models in terms of morphology, ion channel characteristics, synaptic dynamics etc. to generate realistic digital models of different kinds of neurons.  This also shows how to optimize parameters which cannot be obtained from experimental data such as the maximum conductance of each ion channel using different tools from The Blue Brain Project. 