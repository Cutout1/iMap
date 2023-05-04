README and REPO WIP

### File Overview
Working Python Scripts:
- sample_libs.py — Random samples of a standard cell library and printing results of mapping. Modified from original version by Dr. Cunxi Yu to iterate through selection sizes automatically and to always include the _const0_ and _const1_ cells from the 7nm.genlib file for higher rates of successful mappings.
- iMapSequential.py — A few deterministic methods for sequential improvement of standard cell sampling. See below for usage notes.
- iMapSequentialDQN.py — Implementation of a Deep Q Network using Pytorch which can be setup to sequentially attempt to optimize either area or delay of a design.
- sequentialEnv.py — Custom gym environment which interfaces with ABC. Required for iMapSequentialDQN.py

### usage
```bash
python sample_libs.py 100 10 ode.abc.blif 7nm.genlib
# param1: highest number of cells to sample
# param2: number of tests per sample count
# param3: design file
# param4: library file
# Randomly choose 10 cells 10 times, then 20 cells 10 times, etc up to 100 cells 10 times and print area and delay data
```
```bash
python iMapSequential.py
# Runs sequential reduction in delay or area by removing cells from the library. Default behavior is to remove the cell that provides the best improvement each iteration, but this is slow as it must check every cell each iteration. Alternate approaches can be performed using find_best_option_faster and find_better_option functions.
# Command line arguments not yet configured. Variables can be configured in the code:
# goal = "min_area" or "min_delay" to minimize area or delay
# lib_path = path to the genlib file, e.g. "7nm.genlib"
# design_path = path to design, e.g. "ode.abc.blif"
```
```bash
python iMapSequentialDQN.py
# Trains a DQN to perform sequential reduction of area or runtime
# Command line arguments not yet configured. Variables can be configured in the code:
# env = MapEnv(lib_path="7nm.genlib", design_path="ode.abc.blif", goal="min_delay")
# DQN variables that can be adjusted: BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR
```
```bash
python sequentialEnv.py
# Provides an environment which can link DQN to ABC more easily. Instantiate like this:
# env = MapEnv(lib_path="7nm.genlib", design_path="ode.abc.blif", goal="min_delay")
```

### how to setup

- 1. setup ABC in global path (any ABC that you have previously compiled)

```bash
export PATH=your-abc-path:${PATH}
# e.g., export PATH=/home/cunxi/cunxi/abc:${PATH}
```

- 2. setup your python env to try the code.

