import os
import pdb 
import numpy as np
import pandas as pd
from torch import dtype

from tqdm import tqdm 
from sklearn.metrics import f1_score
from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def read_from_file(file_in):
    df = pd.read_csv(file_in, delimiter="\t")
    return df

runs = [
    # "60_InvSqrt", 
    # "60_VAT_1",
    # "80_InvSqrt", 
    # "90_InvSqrt", 
    # "90_InvSqrt_VAT",
    "Adapters_VAtt",
    "Adapters_VAtt_2",
    # "FT_VAtt", 
    # "90_VAT_1", 
    # "FT_90",
    # "FT_90_2", 
    # "90_Head_2", 
    # "100_InvSqrt", 
    # "110_InvSqrt", 
    # "120_InvSqrt", 
] # subtask 1.2

run_id = 15
subtask = "DA_Country"
verbose = True
 

dirname = "/home/bkhmsi/Documents/Projects/NeuralDialectDetector/checkpoints_marbert"
files = [f"{dirname}/MARBERT_{subtask}_{i}/predictions_dev.tsv" for i in runs]

def parse_results(inf):
    with open(inf) as fin:
        dat = [l.strip().split('\t') for l in fin][2:]
        ordd = np.array([x[0] for x in dat], dtype=np.int).argsort()
        datb = np.array([x[1][1:-1].split() for x in dat], dtype=float)[ordd]
        datc = np.array([x[-1] for x in dat], dtype=int)[ordd]
        return datb, datc

dat = [parse_results(k) for k in files]

labels = np.stack([x[1] for x in dat])
probs  = np.stack([x[0] for x in dat]).transpose(1, 0, 2)

subsets = list(powerset(np.arange(len(runs))))[1:]

accs = np.zeros(len(subsets))
f1s = np.zeros(len(subsets))
predictions = []
for i, subset in tqdm(enumerate(subsets), total=len(subsets)):
    probs_subset = probs[:,subset,:].mean(axis=1)
    preds_subset = probs_subset.argmax(-1)
    accs[i] = np.mean(labels[0]==preds_subset)
    f1s[i] = f1_score(labels[0], preds_subset, average="macro")
    predictions += [preds_subset]

    if verbose:
        print("="*20)
        print([runs[j] for j in subset])
        print(f"F1: {f1s[i]} | Acc: {accs[i]}")

idx = np.argmax(f1s)
print(f"Max Acc.: {accs.max()}")
print([runs[j] for j in subsets[idx]])
print(f"F1: {f1s[idx]} | Acc: {accs[idx]}")

if "DA" in subtask:
    dirname = "NADI2021_DEV.1.0/NADI2021_DEV.1.0/Subtask_1.2+2.2_DA"
    basename = "classes_12.txt" if "Country" in subtask else "classes_22.txt"
    gt_name = "subtask12_GOLD.txt" if "Country" in subtask else "subtask22_GOLD.txt"
    subtask_id = 12 if "Country" in subtask else 22
elif "MSA" in subtask:
    dirname = "NADI2021_DEV.1.0/NADI2021_DEV.1.0/Subtask_1.1+2.1_MSA"
    basename = "classes_12.txt" if "Country" in subtask else "classes_22.txt"
    gt_name = "subtask11_GOLD.txt" if "Country" in subtask else "subtask21_GOLD.txt"
    subtask_id = 11 if "Country" in subtask else 21
else:
    raise ValueError(subtask)

with open(os.path.join(dirname, basename), 'r') as fin:
    countries = [x.strip() for x in fin.readlines()]
countries.sort()

with open(f"CairoSquad_subtask{subtask_id}_dev_{run_id}.txt", 'w') as fout:
    fout.write('\n'.join([countries[x] for x in predictions[idx]]))