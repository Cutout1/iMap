README and REPO WIP

### File Overview
Working Python Scripts:
- sample_libs.py — Random samples of a standard cell library and printing results of mapping. Modified from original version by Dr. Cunxi Yu to iterate through selection sizes automatically and to always include the _const0_ and _const1_ cells from the 7nm.genlib file for higher rates of successful mappings.
- iMapSequential.py — A few deterministic methods for sequential improvement of standard cell sampling. See below for usage notes.
- iMapSequentialDQN.py — Implementation of a Deep Q Network using Pytorch which can be setup to sequentially attempt to optimize either area or delay of a design.
- sequentialEnv.py — Custom gym environment which interfaces with ABC. Required for iMapSequentialDQN.py

### usage
```bash
python sample_libs.py 100 ode.abc.blif 7nm.genlib
# it does randomly choose 90 cells out of 7nm.genlib and performance mapping with ABC
```

### how to setup

- 1. setup ABC in global path (any ABC that you have previously compiled)

```bash
export PATH=your-abc-path:${PATH}
# e.g., export PATH=/home/cunxi/cunxi/abc:${PATH}
```

- 2. setup your python env to try the code.

