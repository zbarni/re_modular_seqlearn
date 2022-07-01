# Towards reproducible models of sequence learning: replication and analysis of a modular spiking network with reward-based learning 

This is a repository accompanying the paper
### Zajzon, B., Duarte, R., & Morrison, A.  (2022). Towards reproducible models of sequence learning: replication and analysis of a modular spiking network with reward-based learning

which provides a re-implementation and analysis of the sequence learning model described in Cone &amp; Shouval (2021). 
The repository contains the code and data necessary to run the simulations, generate the datasets and reproduce the figures in the manuscript.


## Installation instructions

### Dependencies

* **Python** 3.8
* [**NEST 2.20.0**](http://www.nest-simulator.org/) - modified version is attached under `nest-simulator-2.20.0-eligibility.zip`. 
You can follow the standard NEST  instructions to install NEST into a directory of your choice. Make sure the environment
variables are loaded properly.
* **FNA (Functional Neural Architectures)** - the library is provided as a compressed file 
under `/libs/fna_v0.2.1_modified.zip`. To install, unpack the archive and follow instructions:

```commandline
$ `pip install -r requirements.txt`
$ `pip install .`
```
* **setuptools** version 52.0.0 or lower
* **numpy** version 1.7.0 or higher 
* **scipy** version 0.12.0 or higher
* **scikit-learn** version 0.18.0 or higher
* **matplotlib** version 1.2.0 or higher

### Installing the neuron and synapse models
The NEST models are found in `/nest_models/`. In the following, we assume that the NEST installation
is located under `/path/to/my/nest/`

To install the neuron model, run:

```commandline
$ cd nest_models/shouval_esyn_neuron_model
$ cmake -Dwith-nest="/path/to/my/nest/bin/nest-config" .
$ make 
$ make install
```

Similarly, to install the synapse model, run:

```commandline
$ cd nest_models/shouval_esyn_synapse_model
$ cmake -Dwith-nest="/path/to/my/nest/bin/nest-config" .
$ make 
$ make install
```

## Running experiments and generating datasets and plotting figures from simulation output
    
Detailed steps on running the experiments are provided in `EXPERIMENTS.md`. 

### Contact

For any questions, please contact any of the authors.

Barna Zajzon - b.zajzon at fz-juelich dot de <br>

