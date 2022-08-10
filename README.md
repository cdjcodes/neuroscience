![](banner.png)
<!-- PROJECT SHIELDS -->
![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/cdjcodes/neuroscience)
![GitHub contributors](https://img.shields.io/github/contributors-anon/cdjcodes/neuroscience)

Personal repository for all Neuroscience related projects. 

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Contents</summary>
  <ol>
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
7. [Blue Brain NeuroM](https://github.com/BlueBrain/NeuroM) 
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