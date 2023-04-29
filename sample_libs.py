import sys
import random
import os
import numpy as np
if len(sys.argv) !=4:
    assert(0), 'usage wrong: please use: python thisfile num_of_gates(int) design.file genlib.file'
# genlib 
genlib_origin = sys.argv[-1]
# design file
design = sys.argv[-2]
# number of design batch
batch_size = 2 
# sample gates
sample_gate = int(sys.argv[-3])



def randlib(genlib_origin, output_genlib, sample=130,sample_feats_fname='samples.csv'):
    with open(genlib_origin) as f:
        lines = f.readlines()
    f.close()
    n_cells = len(lines)
    assert(n_cells > sample)

    randomlist = random.sample(range(3, n_cells-1), sample - 2)
    randomlist.append(1)
    randomlist.append(2)
    #print(randomlist)
    lines_partial = [lines[i] for i in randomlist]

    output_genlib_file = output_genlib + ".genlib"
    output_genlib_feats = output_genlib + ".csv"
    with open(output_genlib_file, 'w') as f:
        for line in lines_partial:
            f.write(line)
    f.close()

    with open(sample_feats_fname, 'a') as f2:
        f2.write(str(randomlist)+'\n')
    f2.close()
    return randomlist

def randlib_batch(batch_size):
    os.system('rm run_batch.sh')
    with open("run_batch.sh", 'a') as f: 
        for i in range(50):
            randlib("7nm.genlib", "newlib_"+str(i))
            abc_cmd = "abc -c \"read %s;read %s; map; ps;\"| grep \"delay\" >> %s &\n" % ("newlib_"+str(i)+".genlib", design, "newlib_"+str(i)+".log")
            print(i,abc_cmd)
            f.write(abc_cmd)
        f.write('wait\n')
    f.close()

def get_map_results(log):
    with open(log) as f:
        lines = f.readlines()
    print(lines)
    assert(len(lines)==1)
    print(lines[0])

import subprocess
from subprocess import PIPE
import re

delay_hist = []
area_hist = []
print(sys.argv)

def sample_test(sample_iter=50, sample_gate=50):
    os.system('rm ' + design+".samples.csv")
    for i in range(sample_iter):
        rand_sample = randlib('7nm.genlib', "newlib_"+str(i), sample_gate, sample_feats_fname=design+".samples.csv")
        abc_cmd = "read %s;read %s; map; write temp.blif; read 7nm_lvt_ff.lib;read -m temp.blif; ps; topo; upsize; dnsize; stime; " % ("newlib_"+str(i)+".genlib", design)
        #abc_cmd = "\"read %s;read %s; map; ps;\"| grep \"delay\" >> %s &\n" % ("newlib_"+str(i)+".genlib", design, "newlib_"+str(i)+".log")
        res = subprocess.check_output(('abc', '-c', abc_cmd))
        match_d = re.search(r"Delay\s*=\s*([\d.]+)\s*ps", str(res))
        match_a = re.search(r"Area\s*=\s*([\d.]+)", str(res))
        if match_d and match_a:
            delay = float(match_d.group(1))
            area = float(match_a.group(1))
        else:
            delay, area = float("NaN"),float("NaN") 
        delay_hist.append(delay)
        area_hist.append(area)
    return np.nanmin(delay_hist), np.nanmin(area_hist), delay_hist, area_hist

x = 10
while x <= 150:
	delay_hist = []
	area_hist = []
	min_d, min_a, d_hist, a_hist = sample_test(100, x)
	print(x, min_d, min_a, np.nanmean(d_hist), np.nanmean(a_hist), len(d_hist), len(a_hist))
	x += 10
	print(delay_hist)
	print(area_hist)
