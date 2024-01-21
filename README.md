# Gaussian Process Regression in Molecular Dynamics

## Overview
This repository contains the implementation of Gaussian process (GP) regression models designed to predict potential energies and forces in molecular dynamics simulations, specifically for water systems. 
The project utilizes atomic descriptors, distances, and Smooth Overlap of Atomic Positions (SOAP) vectors to enhance the precision of simulations.
We also emply Bayesian Optimisation to find the optimal low-energy stable configurations of the system being simulated. 

## Features
- **GP Regression Models**: Implements advanced GP regression techniques to model molecular dynamics.
- **Custom Kernel Design**: Tailored kernels optimized for specific characteristics of water systems.
- **Hyperparameter Tuning**: Fine-tuning of model parameters for optimal performance.
- **Bayesian Optimization**: Utilizes Bayesian methods to efficiently search for stable, low-energy molecular conformations.

## Installation
pip install -r requirements.txt

## Usage
python main.py

## Data
Obtained directly from running ASE simulations of Water molecules. 

## Full Report
For more detailed information, see the [Full Project Report](/L48_project.pdf).
