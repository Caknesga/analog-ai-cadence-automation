# Analog-AI-Cadence-Automation

## Overview

This is a research repository for developing, testing, and validating neural networks that are intended to be implemented on custom analog and mixed-signal integrated circuits. The repository bridges the gap between machine learning software and analog hardware by providing an automated workflow from neural network inference to Cadence circuit simulation.

The main objective of this project is to verify whether a trained neural network can be accurately mapped onto analog hardware while maintaining its prediction accuracy and electrical performance.

---

## Features

* Neural network training and inference in Python
* Automated Cadence OCEAN simulation pipeline
* Analog/mixed-signal circuit parameter extraction
* Software-to-hardware verification
* Performance comparison between software and analog implementation
* Batch simulation and result evaluation
* Automated data processing and visualization

---

## Workflow

The repository follows the development flow shown below:

```
Dataset
      │
      ▼
Data Preprocessing
      │
      ▼
Neural Network Training
      │
      ▼
Model Evaluation
      │
      ▼
Parameter Mapping
      │
      ▼
Automatic Cadence OCEAN Script Generation
      │
      ▼
Cadence Analog Circuit Simulation
      │
      ▼
Simulation Result Extraction
      │
      ▼
Performance Comparison
```

---

## Purpose

This repository was developed to automate one of the most time-consuming tasks during analog AI hardware development.

Instead of manually modifying circuit parameters and executing Cadence simulations, the complete workflow is automated. Python scripts generate the required parameters, execute Cadence OCEAN simulations, extract the simulation results, and compare them with the original neural network predictions.

This significantly accelerates the hardware validation process and enables rapid testing of different circuit configurations.

---

## Main Components

### Neural Network

* Model development
* Training
* Validation
* Accuracy evaluation
* Parameter export

---

### Mapping Layer

Converts the trained neural network parameters into hardware-compatible values that can be used inside analog circuit simulations.

Examples include

* transistor bias values
* current sources
* voltage references
* programmable weights

depending on the target hardware architecture.

---

### Cadence Automation

The repository automatically communicates with Cadence Virtuoso using OCEAN scripts.

Supported tasks include

* automatic netlist generation
* parameter modification
* simulation execution
* result extraction
* batch simulations

This removes nearly all manual interaction with Cadence during hardware verification.

---

### Verification

Simulation outputs are automatically compared against the software neural network.

Metrics include

* prediction accuracy
* inference error
* output deviation
* circuit performance

This allows direct validation of the analog implementation.

---

## Applications

The framework is suitable for

* Analog Neural Networks
* Edge AI accelerators
* Mixed-Signal AI systems
* Custom AI Integrated Circuits
* Neuromorphic Computing
* Research in Hardware AI

---

## Current Status

This repository is actively used as a research platform for developing analog neural network hardware and automating the verification process between software models and Cadence-based circuit simulations. New mapping strategies, automation features, and hardware validation methods are continuously integrated as the project evolves.

---

## License

This repository is intended for research and educational purposes.

## Additional Notes for myself 

If you see:

conda: command not found

You need to initialize Conda for your shell once.

Run:

source ~/miniconda3/etc/profile.d/conda.sh


Then:

conda activate analog_ai


#Steps to run ocean file wiht python that you created from your schmeatic maestro of your analog circuit and that you saved in the SERVER

1- ssh -X user@server 

2- source ~/miniconda3/etc/profile.d/conda.sh

3- conda activate analog_ai

4-source cadence (get *WARNING* (DISPLAY "<not defined>") ERROR but not problem , 
you just wont see the render of cadence)

5- cd repository

6- Start the python file that runs source the ocean atuoamtically (if not - ocean -nograph -restore cadence/ocean/run_one_layer.ocn)

7- You hav ethe simualtions result as .csv data
