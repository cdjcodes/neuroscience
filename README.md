![](banner.png)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://neurosimlabcdj.streamlit.app/)
[![tests](https://github.com/cdjcodes/neuroscience/actions/workflows/tests.yml/badge.svg)](https://github.com/cdjcodes/neuroscience/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/cdjcodes/neuroscience)

# NeuroSim Lab: virtual experiments on real neurons

**Try it in your browser: [neurosimlabcdj.streamlit.app](https://neurosimlabcdj.streamlit.app/)**

Pick a real neuron from a public database, see it in 3D, inject current into it, and compare the simulated response with what the real cell did in the lab.

- **Real cells:** 300,000+ reconstructed neurons from [NeuroMorpho.org](https://neuromorpho.org), and 667 neurons from the [Allen Cell Types Database](https://celltypes.brain-map.org) that were both recorded with patch-clamp and reconstructed.
- **Real simulations:** every branch of the neuron is simulated with [NEURON](https://nrn.readthedocs.io), the standard simulator in computational neuroscience.
- **Checked against reality:** for Allen cells, the app overlays the recording on each model and compares them feature by feature.

This repository also contains my notebooks from EPFL's [Simulation Neuroscience](https://www.edx.org/course/simulation-neuroscience) course, which the app grew out of. See [Course notebooks](#course-notebooks).

## Contents

- [What you can do in the app](#what-you-can-do-in-the-app)
- [Run it on your computer](#run-it-on-your-computer)
- [Use the Python package](#use-the-python-package)
- [How the models work](#how-the-models-work)
- [Deploy your own copy](#deploy-your-own-copy)
- [Course notebooks](#course-notebooks)
- [Credits and data sources](#credits-and-data-sources)

## What you can do in the app

1. **Choose a neuron.** Browse Allen cells by species, brain region and dendrite type, or search NeuroMorpho.org by species, region, cell type or lab. You can also upload your own `.swc` reconstruction, or use the ball-and-stick teaching model.
2. **Explore its shape** (*Morphology* tab). Rotate it in 3D and see its total length, branch points, Sholl analysis and branch statistics.
3. **Run an experiment** (*Current clamp* tab). Inject a current step into the soma and watch the voltage at the soma, a dendrite tip and the axon, plus a 3D animation of the voltage spreading through the cell.
4. **Measure its input-output curve** (*f-I curve* tab). See firing rate against injected current, and find the rheobase (the smallest current that makes it fire).
5. **Compare with the real recording** (*Compare with recording* tab, Allen cells only). Replay any current step the Allen Institute recorded and overlay the real voltage trace with each model. The tab includes a close-up of the first spike, a table of electrophysiology features and the recorded vs. simulated f-I curves.

Use the sidebar to set the stimulus and choose the membrane model:

| Model | Available for | Best at |
| --- | --- | --- |
| **Allen biophysical model** | 236 Allen cells | Realistic spikes. The Allen Institute fitted it to this cell using cortical ion channels. |
| **Hodgkin-Huxley fitted here** | Any Allen cell (one click, about 1 minute) | Resting potential, input resistance and rheobase. Spikes are too wide. |
| **Generic Hodgkin-Huxley** | Every neuron | Exploring: every parameter has a slider. |

## Run it on your computer

You need Python 3.10 or newer. A C compiler is needed only for the Allen biophysical models (on macOS: `xcode-select --install`).

```bash
git clone https://github.com/cdjcodes/neuroscience.git
cd neuroscience
python -m venv .venv && source .venv/bin/activate
pip install -e ".[app,dev]"

streamlit run streamlit_app.py      # opens http://localhost:8501
```

To run the tests:

```bash
pytest                    # full suite
pytest -m "not network"   # skip tests that call NeuroMorpho.org and the Allen API
```

Downloads, compiled ion channels and model fits are cached in `~/.cache/neurosim/`.

## Use the Python package

Everything the app does is available from Python through the `neurosim` package.

```python
from neurosim import ephys, morphology, simulation as sim
from neurosim.sources import AllenCellTypesClient

allen = AllenCellTypesClient()
cell = allen.get(485909730)                        # a mouse layer-5 pyramidal neuron
morph = morphology.load(allen.fetch_swc(cell))
print(morphology.summary(morph))                   # total length, branch points, ...

# Replay one of the recorded current steps on the Allen Institute's model of this cell
sweep = next(s for s in allen.sweeps(cell.specimen_id) if s.number == 35)
recorded = allen.trace(cell, sweep)                # streams ~3 MB, not the whole recording file
model = sim.AllenPerisomaticCell(morph, allen.perisomatic_model(cell))
result = model.run(sim.CurrentClamp(amp=sweep.amplitude_pa / 1000, delay=100, dur=1000, tstop=1300))

print(ephys.compare(
    ephys.features(recorded.t, recorded.v, 0, 1000),
    ephys.features(result.t - 100, result.v["soma"], 0, 1000),
))                                                 # 17 vs 17 spikes, 1.08 vs 1.14 ms spike width, ...
```

| Module | What it does |
| --- | --- |
| `neurosim.sources` | Search and download from NeuroMorpho.org and the Allen Cell Types Database (cells, recordings, published models). |
| `neurosim.morphology` | Load reconstructions (repairing common file problems) and measure them with [NeuroM](https://github.com/openbraininstitute/NeuroM). |
| `neurosim.simulation` | Build cells in NEURON and run current-clamp experiments, f-I curves and rheobase searches. |
| `neurosim.channels` | Compile the cortical ion-channel models used by the Allen biophysical models. |
| `neurosim.ephys` | Measure spikes and other electrophysiology features with [eFEL](https://github.com/openbraininstitute/eFEL). |
| `neurosim.fitting` | Fit a Hodgkin-Huxley model to a recorded Allen cell. |
| `neurosim.viz` | The Plotly figures used by the app. |

## How the models work

Every model uses the neuron's real 3D shape: each branch becomes a NEURON section with its measured lengths and diameters.

**Allen biophysical models** (`AllenPerisomaticCell`). For 236 reconstructed cells, the Allen Institute published a model fitted to that cell's recordings. It has cortical ion channels in the soma (sodium, several potassium currents, calcium and Ih), passive dendrites and a short axon stub, simulated at 34 °C. The app rebuilds each model following Allen's recipe. On the sweeps they were fitted to, these models closely reproduce the spike count, timing and spike shape. Near firing threshold they can differ more from the recording.
- **Channel files:** the 16 channel definitions live in [`neurosim/mechanisms/allen_perisomatic`](neurosim/mechanisms/allen_perisomatic). They were reviewed and pinned, so no code is downloaded at runtime; that folder's README lists their sources and checksums. They are compiled once, on first use.
- **Voltage offset:** Allen fitted these models to recordings corrected for the junction potential, so the app shifts their output by +14 mV to line up with the raw recordings.

**Hodgkin-Huxley models** (`MorphologyCell`). These use the classic squid-axon channels in the soma and axon, passive dendrites, and an axon initial segment where spikes start (a synthetic one is added if the reconstruction has no axon). They run for any neuron. Their spikes are wider and taller than real cortical spikes.

**Fitting** (`neurosim.fitting`). This tunes a Hodgkin-Huxley model to an Allen cell in two stages:
1. **Passive properties:** the leak and dendritic capacitance are fitted to the measured resting potential, input resistance and time constant. Extra dendritic capacitance stands in for spines, which reconstructions leave out.
2. **Firing:** the sodium density is fitted so the model starts firing at the recorded rheobase. Potassium density and channel speed are then fitted to the recorded f-I curve.

Fits match the passive properties and rheobase closely and firing rates reasonably well. The app flags cells whose reconstruction looks incomplete, because those need an implausibly high capacitance.

## Deploy your own copy

The live app runs free on [Streamlit Community Cloud](https://streamlit.io/cloud) and redeploys on every push to `main`. To host your own:

1. Fork this repository.
2. At [share.streamlit.io](https://share.streamlit.io), choose **Create app → Deploy a public app from GitHub**.
3. Select your fork, branch `main` and file `streamlit_app.py`. Under **Advanced settings**, pick Python 3.12, then click **Deploy**.

Python packages come from `requirements.txt`. `packages.txt` installs the C compiler the ion channels need.

## Course notebooks

The [`Simulation Neuroscience`](Simulation%20Neuroscience) folder holds my exercises from EPFL's *Simulation Neuroscience* MOOC on edX. The course covers how to reconstruct a neuron digitally and simulate how neurons connect.

| Folder | Topics |
| --- | --- |
| Accessing and Manipulating Neuroscience Data | Getting data from the Allen Cell Types Database; organising it and its metadata (MINDS) in Blue Brain Nexus; a content-based recommender built on the knowledge graph |
| Neuron Morphologies | Loading, visualising and measuring neuron shapes, and statistics across neurons |
| Ball and Stick Model | Building a neuron from a Hodgkin-Huxley soma and a dendrite; synapses, EPSPs and IPSPs; electrical synapses |
| Neuron Simulations | Constraining models with experimental data; optimising channel conductances with BluePyOpt |

The notebooks are kept as a record of the course and may not all run today. The Blue Brain Nexus sandbox they use has gone offline, and several Blue Brain tools have since moved to the [Open Brain Institute](https://github.com/openbraininstitute) or been archived. The `neurosim` package is the maintained version of the morphology and simulation material.

## Credits and data sources

- **Data:** [NeuroMorpho.org](https://neuromorpho.org); [Allen Cell Types Database](https://celltypes.brain-map.org) © Allen Institute for Brain Science, used under its [terms of use](https://alleninstitute.org/terms-of-use/).
- **Tools:** [NEURON](https://github.com/neuronsimulator/nrn), [NeuroM](https://github.com/openbraininstitute/NeuroM), [eFEL](https://github.com/openbraininstitute/eFEL), [Streamlit](https://streamlit.io), [Plotly](https://plotly.com/python/).
- **Course:** [Simulation Neuroscience](https://www.edx.org/course/simulation-neuroscience) (EPFL / Blue Brain Project) and its [exercise repository](https://github.com/BlueBrain/MOOC-neurons-and-synapses-2017).
- **Further reading:** [AllenSDK](https://allensdk.readthedocs.io/en/latest/), [BluePyOpt](https://github.com/openbraininstitute/BluePyOpt), [BluePyMM](https://github.com/BlueBrain/BluePyMM), [NMODL](https://github.com/BlueBrain/nmodl) (archived), [MouseLight project](https://www.janelia.org/project-team/mouselight).
