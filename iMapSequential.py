import sys
import os
import subprocess
from subprocess import PIPE
import re

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np

def check_selection(selection, source_lib, design_path):
    with open("newlib.genlib", 'w') as f:
        for x in range(len(selection)): #dim may need to be 1
            if selection[x]:
                f.write(source_lib[x+1])
    f.close()
    
    abc_cmd = "read %s;read %s; map; write temp.blif; read 7nm_lvt_ff.lib;read -m temp.blif; ps; topo; upsize; dnsize; stime; " % ("newlib.genlib", design_path)
    res = subprocess.check_output(('abc', '-c', abc_cmd))
    
    match_d = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", str(res))
    match_a = re.search(r"Area\s*=\s*([\d.]+)", str(res))

    delay = 0
    area = 0

    if match_d and match_a:
        delay = float(match_d.group(1))
        area = float(match_a.group(1))
    else:
        delay, area = float("NaN"),float("NaN")

    return delay, area


def compute_reward(delay, area, goal):
    if goal == "min_area":
        if delay == float("NaN") or area == float("NaN"):
            return -20000
        else:
            return -1*area
    else:
        if delay == float("NaN") or area == float("NaN"):
            return -2000
        else:
            return -1*delay

def find_best_option(selection, source_lib, design_path): #685.07 delay
    best_selection = selection
    best_delay, best_area = check_selection(best_selection, source_lib, design_path)
    for x in range(len(selection)):
        if selection[x] == 0:
            continue
        new_selection = np.copy(selection)
        new_selection[x] = 0
        
        delay, area = check_selection(new_selection, source_lib, design_path)
        print(str(x) + " Delay: " + str(delay) + "    Area: " + str(area))

        if delay != float("NaN") and delay < best_delay:
            best_selection = new_selection
            best_delay = delay
            area = area

    return best_selection, best_delay, best_area

def find_best_option_faster(selection, source_lib, design_path, skip_list):
    best_selection = selection
    best_delay, best_area = check_selection(best_selection, source_lib, design_path)
    original_delay = best_delay
    for x in range(len(selection)):
        if skip_list[x] == 1:
            continue
        new_selection = np.copy(selection)
        new_selection[x] = 0
        
        delay, area = check_selection(new_selection, source_lib, design_path)
        print(str(x) + " Delay: " + str(delay) + "    Area: " + str(area))

        if delay != float("NaN"):
            if delay < best_delay:
                best_selection = new_selection
                best_delay = delay
                area = area
                skip_list[x] == 1
            elif delay > original_delay:
                skip_list[x] = 1
        else:
            skip_list[x] = 1

    return best_selection, best_delay, best_area

def find_better_option(selection, source_lib, design_path): #700.78
    best_selection = selection
    best_delay, best_area = check_selection(best_selection, source_lib, design_path)
    for x in range(len(selection)):
        if selection[x] == 0:
            continue
        new_selection = np.copy(selection)
        new_selection[x] = 0
        
        delay, area = check_selection(new_selection, source_lib, design_path)
        print(str(x) + " Delay: " + str(delay) + "    Area: " + str(area))

        if delay != float("NaN") and delay < best_delay:
            best_selection = new_selection
            best_delay = delay
            area = area
            break

    return best_selection, best_delay, best_area




goal = "min_area"

lib_path="7nm.genlib"

design_path="ode.abc.blif"



with open(lib_path) as f:
    source_lib = f.readlines()
f.close()

n_cells = len(source_lib) - 1

print("Number of cells in source library: " + str(n_cells))



print(check_selection(selection, source_lib, design_path))

"""
selection = np.ones(n_cells)
skip_list = np.zeros(n_cells)

while(True):
    new_selection, new_delay, new_area = find_best_option_faster(selection, source_lib, design_path, skip_list)
    if np.array_equal(selection, new_selection):
        break
    selection = new_selection

print("Best selection found: " + str(selection))

for x in range(5):
    delay, area = check_selection(selection, source_lib, design_path)
    print("Delay: " + str(delay) + "    Area: " + str(area))
"""







