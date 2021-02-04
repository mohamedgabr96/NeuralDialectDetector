import os
import pdb 
import numpy as np
import pandas as pd
from torch import dtype

from tqdm import tqdm 
from sklearn.metrics import f1_score
from itertools import chain, combinations

def read_from_file(file_in):
    df = pd.read_csv(file_in, delimiter="\t")
    return df

runs = [
    "Adapters_VAtt",
    "FT_VAtt", 
    "FT_90",
    "FT_90_2", 
] # subtask 1.2

# runs = [
#     "90_FT_VAtt",
#     "90_FT",
#     "90_FT_2"
# ] # subtask 2.2

# runs = [
#     "110_Adapters_VAtt",
#     "110_FT_VAtt",
#     "110_FT",
#     "110_FT_2"
# ] # subtask 1.1

run_id = 2
subtask = "DA_Country"
# subtask = "DA_Province"
# subtask = "MSA_Country"
# subtask = "MSA_Province"
dset = "test"
 

dirname = "/home/bkhmsi/Documents/Projects/NeuralDialectDetector/checkpoints_marbert"
files = [f"{dirname}/MARBERT_{subtask}_{i}/predictions_{dset}.tsv" for i in runs]

def parse_results(inf):
    with open(inf) as fin:
        dat = [l.strip().split('\t') for l in fin][2:]
        ordd = np.array([x[0] for x in dat], dtype=np.int).argsort()
        datb = np.array([x[1][1:-1].split() for x in dat], dtype=float)[ordd]
        return datb

dat = [parse_results(k) for k in files]
probs = np.stack([x for x in dat]).transpose(1, 0, 2)

probs_subset = probs.prod(axis=1)
preds_subset = probs_subset.argmax(-1)

if "DA" in subtask:
    dirname = "NADI2021_DEV.1.0/NADI2021_DEV.1.0/Subtask_1.2+2.2_DA"
    basename = "classes_12.txt" if "Country" in subtask else "classes_22.txt"
    subtask_id = 12 if "Country" in subtask else 22
elif "MSA" in subtask:
    dirname = "NADI2021_DEV.1.0/NADI2021_DEV.1.0/Subtask_1.1+2.1_MSA"
    basename = "classes_12.txt" if "Country" in subtask else "classes_22.txt"
    subtask_id = 11 if "Country" in subtask else 21
else:
    raise ValueError(subtask)

with open(os.path.join(dirname, basename), 'r') as fin:
    countries = [x.strip() for x in fin.readlines()]
countries.sort()

with open(f"CairoSquad_subtask{subtask_id}_{dset}_{run_id}.txt", 'w') as fout:
    fout.write('\n'.join([countries[x] for x in preds_subset]))