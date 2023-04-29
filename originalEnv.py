from collections import defaultdict
from typing import Optional

import random

import sys
import os
import subprocess
from subprocess import PIPE
import re
import numpy as np
import torch
import tqdm
from tensordict.nn import TensorDictModule
from tensordict.tensordict import TensorDict, TensorDictBase
from torch import nn

from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec, BinaryDiscreteTensorSpec
from torchrl.envs import EnvBase
from torchrl.envs.utils import check_env_specs, step_mdp
import torchrl

# in env constructor take the path to the genlib and read and store it
# also store the path to the design
# also take in goal (minimize area, delay, etc)

def _step(self, tensordict):
    
    selection = tensordict["action"]
    
    with open("newlib.genlib", 'w') as f:
        for x in range(selection.size(dim=0)): #dim may need to be 1
            if selection[x]:
                f.write(self.source_lib[x+1])
    f.close()
    
    abc_cmd = "read %s;read %s; map; write temp.blif; read 7nm_lvt_ff.lib;read -m temp.blif; ps; topo; upsize; dnsize; stime; " % ("newlib.genlib", self.design_path)
    res = subprocess.check_output(('abc', '-c', abc_cmd))
    
    match_d = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", str(res))
    match_a = re.search(r"Area\s*=\s*([\d.]+)", str(res))
    reward = 0
    delay = 0
    area = 0
    if match_d and match_a:
        delay = float(match_d.group(1))
        area = float(match_a.group(1))
        
        if self.goal == "min_area":
            reward = 10000 - area
        if self.goal == "min_delay":
            reward = 1000 - delay
    else:
        delay, area = float("NaN"),float("NaN")
        if self.goal == "min_area":
            reward = -10000
        if self.goal == "min_delay":
            reward = -1000

    print("Delay: " + str(delay) + "    Area: " + str(area))

    reward = torch.tensor(reward, dtype=torch.float32)
    
    done = torch.zeros_like(reward, dtype=torch.bool)
    
    out = TensorDict(
        {
            "next": {
                "reward": reward,
                "done": done,
            }
        },
        tensordict.shape,
    )
    return out


def _reset(self, tensordict):
    if tensordict is None or tensordict.is_empty():
        td = TensorDict({},[])
        return td
    return tensordict

def _make_spec(self): # need to addjust observation_spec and input_spec
    self.observation_spec = CompositeSpec(
        shape=(),
    )

    self.input_spec = CompositeSpec(
        shape=(),
    )
    self.action_spec = BinaryDiscreteTensorSpec(
        n=self.n_cells,
        dtype=torch.bool,
    )
    self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))


def _set_seed(self, seed):
    return

class MapEnv(EnvBase):
    batch_locked = False

    def __init__(self, lib_path, design_path, goal, device, batch_size):
        super().__init__(device=device, batch_size=batch_size)

        self.design_path = design_path
        self.goal = goal
        
        with open(lib_path) as f:
            self.source_lib = f.readlines()
        f.close()
        
        self.n_cells = len(self.source_lib) - 1

        print("Number of cells in source library: " + str(self.n_cells))
        
        self._make_spec()

    _make_spec = _make_spec
    
    _reset = _reset
    _step = _step
    _set_seed = _set_seed







env = MapEnv(lib_path="7nm.genlib", design_path="ode.abc.blif", goal="min_area", device="cpu", batch_size=[])
#check_env_specs(env)

#results = env._step(env.rand_step())

#print(results["next"]["reward"])

env.rollout(max_steps=100)














