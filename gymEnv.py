import gym
from gym import spaces

import numpy as np
import sys
import os
import subprocess
from subprocess import PIPE
import re

class MapEnv(gym.Env):

    def __init__(self, lib_path, design_path, goal):
        super(MapEnv, self).__init__()

        self.design_path = design_path
        self.goal = goal
        
        with open(lib_path) as f:
            self.source_lib = f.readlines()
        f.close()
        
        self.n_cells = len(self.source_lib) - 1

        print("Number of cells in source library: " + str(self.n_cells))

        # action space definition
        self.action_space = spaces.MultiDiscrete(np.ones(self.n_cells)*2)
        # observation space definition
        self.observation_space = spaces.MultiDiscrete(np.ones(self.n_cells)*2)


    # On resetting, the observation will be the full library selected. Could make random later
    def reset(self):
        obs = np.ones(self.n_cells)
        return obs

    def step(self, action):
        action = action[0][0]
        # create the new library based on the cells selected by the action
        with open("newlib.genlib", 'w') as f:
            for x in range(len(action)):
                if action[x]:
                    f.write(self.source_lib[x+1])
        f.close()

        # run the ABC command and retreive results
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

        done = False

        observation = action.clone()

        return observation, reward, done
