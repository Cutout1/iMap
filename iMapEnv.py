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

        self.state = np.ones(self.n_cells, dtype=bool)
        self.num_steps = 0

        abc_cmd = "read %s;read %s; map; write temp.blif; read 7nm_lvt_ff.lib;read -m temp.blif; ps; topo; upsize; dnsize; stime; " % (lib_path, self.design_path)
        res = subprocess.check_output(('abc', '-c', abc_cmd))
        
        self.base_delay = float(re.search(r"Delay\s*=\s*([\d.]+)\s*ps", str(res)).group(1))
        self.base_area = float(re.search(r"Area\s*=\s*([\d.]+)", str(res)).group(1))

        #self.previous_area = self.base_area
        #self.previous_delay = self.base_delay

        # action space definition
        self.action_space = spaces.Discrete(self.n_cells)
        # observation space definition
        self.observation_space = spaces.MultiDiscrete(np.ones(self.n_cells)*2)


    # On resetting, the observation will be the full library selected. Could make random later
    def reset(self):
        self.state = np.ones(self.n_cells, dtype=bool)
        self.num_steps = 0
        self.previous_area = self.base_area
        self.previous_delay = self.base_delay
        return np.copy(self.state)

    def step(self, action):
        self.num_steps += 1
        self.state = action
        # create the new library based on the cells selected by the action
        with open("newlib.genlib", 'w') as f:
            for x in range(self.n_cells):
                if self.state[0][0][x]:
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
        done = False
        if match_d and match_a:
            delay = float(match_d.group(1))
            area = float(match_a.group(1))
            
            if self.goal == "min_area":
                #reward = self.previous_area - area
                reward = self.base_area - area
            if self.goal == "min_delay":
                #reward = self.previous_delay - delay
                reward = self.base_delay - delay
        else:
            done = True
            delay, area = float("NaN"),float("NaN")
            reward = 0

        print("Delay: " + str(delay) + "    Area: " + str(area))

        observation = np.copy(self.state)

        if self.num_steps > self.n_cells:
            done = True

        self.previous_area = area
        self.previous_delay = delay

        return observation, reward, done
