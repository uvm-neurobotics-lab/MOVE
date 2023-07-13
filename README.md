# Many-objective Optimization via Voting for Elites (MOVE)
By Jackson Dean and Nick Cheney

[Paper](https://arxiv.org/abs/2307.02661)


## Setup
Create a conda environment using environment.yml:

`conda env create -f environment.yml`


---
## Running

### Run with default configuration and target
```python move.py```


### Run with custom configuration
```ptyhon move.py -c <path-to-config.json>```


### Run with command line args
```python move.py <args>```


```
options
  -h, --help            show this help message and exit
  --config CONFIG, -c CONFIG
                        Path to config json file (default: move.json).
  --generations GENERATIONS, -g GENERATIONS
                        Number of generations to run (default: 1000).
  --population POPULATION, -p POPULATION
                        Population size.
  --output OUTPUT, -o OUTPUT
                        Output directory.
  -t TARGET, --target TARGET
                        Target image.
  -v, --verbose         Print verbose output (default: False).
  -d DEVICE, --device DEVICE
                        Device to run on (default: cuda:0).
  -sgd, --sgd           Use SGD to update weights (default: True).
  -ff NUM_FOURIER_FEATURES, --num_fourier_features NUM_FOURIER_FEATURES
                        Number of fourier features (default: 8).
  -hn NUM_HIDDEN_NODES, --num_hidden_nodes NUM_HIDDEN_NODES
                        Number of hidden nodes at initialization (default: 8).

```

### SGD and Fourier Features
The default configuration of MOVE uses fourier features as inputs (see [here](https://bmild.github.io/fourfeat/)) and SGD to update CPPN weights.


