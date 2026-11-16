![](banner.png)
<!-- PROJECT SHIELDS -->
![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)
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

`neurosim` is a Python package that turns the course notebooks below into reusable code for running virtual experiments on **real neurons from online databases**:

| Module | What it does | Built on |
| --- | --- | --- |
| `neurosim.sources` | Search [NeuroMorpho.org](https://neuromorpho.org) (300k+ reconstructed neurons) by species, brain region, cell type, lab, etc. and download their reconstructions (cached locally) | NeuroMorpho REST API |
| `neurosim.morphology` | Load reconstructions (repairing the common soma problems in database files), compute morphometrics, Sholl analysis, distributions, 3D geometry for plotting | [NeuroM](https://github.com/openbraininstitute/NeuroM) |
| `neurosim.simulation` | Current-clamp experiments, f-I curves and rheobase, on the classic ball-and-stick model or on a full multi-compartment model built from any downloaded neuron | [NEURON](https://nrn.readthedocs.io) |

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

### Model notes

The multi-compartment model is generic, **not fitted to the individual cell**: Hodgkin-Huxley channels in the soma and axon, passive dendrites, and an axon initial segment (AIS) with a high Na/K density where spikes start (a synthetic AIS is added if the reconstruction has no axon). Without the AIS, the large passive dendritic tree loads the soma so heavily that squid HH channels fire one spike and then block. All parameters are in `simulation.Biophysics`.

### Roadmap

1. ✅ Core library: database access, morphometrics, simulations
2. ⬜ Web app (Streamlit): search the database, 3D viewer, run experiments interactively
3. ⬜ Real electrophysiology: compare simulations against Allen Cell Types recordings (eFEL features) and fit models to them
4. ⬜ Deploy online

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