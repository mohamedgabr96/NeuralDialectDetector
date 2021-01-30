import os 
import numpy as np
import json
import pandas as pd
from scipy.sparse import base
from sklearn.metrics import f1_score

def read_from_file(file_in):
    df = pd.read_csv(file_in, delimiter="\t")
    return df

# indices_list = ["60_InvSqrt", "90_VAT_1", "90_InvSqrt_VAT"] # subtask 1.2
indices_list = ["60", "90_VAT"] # subtask 2.2
indices_list = ["60", "90_VAT_1", "90", "110"] # subtask 1.1
indices_list = ["60", "100", "110", "90_VAT", "90_VAT_1"] # subtask 2.1

subtask = "MSA_Province"
run_id = 1

dirname = "/home/bkhmsi/Documents/Projects/NeuralDialectDetector/checkpoints_marbert"
files_in = [f"{dirname}/MARBERT_{subtask}_{i}/predictions_test.tsv" for i in indices_list]

df_list = []
for i in files_in:
    df_list.append(read_from_file(i))

df_to_join = df_list[0].set_index("Sentence Index")
col_names = ["SoftMaxes"]
df_to_join[f"SoftMaxes"] = df_to_join[f"SoftMaxes"].apply(lambda x: list(map(float, x[1:-1].split())))

for index, dff in enumerate(df_list[1:]):
    df_to_join = df_to_join.join(dff.set_index("Sentence Index"), how="inner", rsuffix=str(index))
    df_to_join[f"SoftMaxes{index}"] = df_to_join[f"SoftMaxes{index}"].apply(lambda x: list(map(float,x[1:-1].split())))
    col_names.append(f"SoftMaxes{index}")

col = df_to_join.loc[:, col_names]

df_to_join["Softmax averaged"] = col.apply(lambda x: np.mean([x[i] for i in col_names], axis=0), axis=1)
df_to_join["Argmaxed average"]= df_to_join["Softmax averaged"].apply(lambda x: np.argmax(x))

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

predictions = []
for i in range(len(df_to_join)):
    predictions += [df_to_join.loc[i][f"Argmaxed average"]]

with open(f"CairoSquad_subtask{subtask_id}_test_{run_id}.txt", 'w') as fout:
    fout.write('\n'.join([countries[x] for x in predictions]))
