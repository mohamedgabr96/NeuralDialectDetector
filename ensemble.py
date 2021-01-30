import os 
import numpy as np
import json
import pandas as pd
from scipy.sparse import base
from sklearn.metrics import f1_score
from tqdm import tqdm 

def read_from_file(file_in):
    df = pd.read_csv(file_in, delimiter="\t")
    return df

# indices_list = [90, 100, 110, "100_VAT"]
# indices_list = [90, 110]
# indices_list = [90, 100]
# indices_list = [90, 100, 110, "100_VAT_1"]
indices_list = ["60_InvSqrt", "90_VAT_1", "90_InvSqrt_VAT", "FT_90", "90_InvSqrt", "FT_90_2", "90_Head_2", "100_InvSqrt", "110_InvSqrt", "120_InvSqrt", "80_InvSqrt", "60_VAT_1"] # subtask 1.2
# indices_list = ["60", "90_VAT"] # subtask 2.2
# indices_list = ["60", "90_VAT_1", "90", "110"] # subtask 1.1
# indices_list = ["60", "100", "110", "90_VAT", "90_VAT_1"] # subtask 2.1

# indices_list = ["90", "100"]
subtask = "DA_Country"
run_id = 10

dirname = "/home/bkhmsi/Documents/Projects/NeuralDialectDetector/checkpoints_marbert"
files_in_all = [f"{dirname}/MARBERT_{subtask}_{i}/predictions_dev.tsv" for i in indices_list]

# files_in_all = [
#     f"{dirname}/MARBERT_{subtask}_60/predictions_dev.tsv",
#     f"{dirname}/MARBERT_{subtask}_90_VAT/predictions_dev.tsv",
#     f"{dirname}/MARBERT_{subtask}_90_VAT_1/predictions_dev.tsv",
#     f"{dirname}/MARBERT_{subtask}_90/predictions_dev.tsv",
#     f"{dirname}/MARBERT_{subtask}_100/predictions_dev.tsv",
#     f"{dirname}/MARBERT_{subtask}_110/predictions_dev.tsv",
# ]

from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

# files_in_all = powerset(files_in)

all_f1s = []
all_accs = []
all_combs = []


df_list_all = []
for i in files_in_all:
    df_list_all.append(read_from_file(i))

for df_list in tqdm(list(powerset(df_list_all))[1:]):

    df_to_join = df_list[0].set_index("Sentence Index")
    col_names = ["SoftMaxes"]
    df_to_join[f"SoftMaxes"] = df_to_join[f"SoftMaxes"].apply(lambda x: list(map(float, x[1:-1].split())))
    df_to_join[f"Argmaxed"]= df_to_join[f"SoftMaxes"].apply(lambda x: np.argmax(x))

    f1_0 = f1_score(df_to_join["Labels"].tolist(), df_to_join[f"Argmaxed"].tolist(), average="macro")
    df_to_join[f"isCorrect"] = df_to_join[f"Argmaxed"] == df_to_join["Labels"]
    acc_0 = df_to_join[f"isCorrect"].mean()
    # print(f"F1 {f1_0} | Acc {acc_0}")


    for index, dff in enumerate(df_list[1:]):
        df_to_join = df_to_join.join(dff.set_index("Sentence Index"), how="inner", rsuffix=str(index))
        df_to_join[f"SoftMaxes{index}"] = df_to_join[f"SoftMaxes{index}"].apply(lambda x: list(map(float,x[1:-1].split())))
        df_to_join[f"Argmaxed{index}"]= df_to_join[f"SoftMaxes{index}"].apply(lambda x: np.argmax(x))
        df_to_join[f"isCorrect{index}"] = df_to_join[f"Argmaxed{index}"] == df_to_join["Labels"]
        acc_i = df_to_join[f"isCorrect{index}"].mean()
        f1_i = f1_score(df_to_join["Labels"].tolist(), df_to_join[f"Argmaxed{index}"].tolist(), average="macro")
        # print(f"F1 {f1_i} | Acc {acc_i}")

        col_names.append(f"SoftMaxes{index}")

    col = df_to_join.loc[:, col_names]

    df_to_join["Softmax averaged"] = col.apply(lambda x: np.mean([x[i] for i in col_names], axis=0), axis=1)
    df_to_join["Argmaxed average"]= df_to_join["Softmax averaged"].apply(lambda x: np.argmax(x))
    df_to_join["isCorrect average"] = df_to_join["Argmaxed average"] == df_to_join["Labels"]
    acc_all = df_to_join["isCorrect average"].mean()
    f1_all = f1_score(df_to_join["Labels"].tolist(), df_to_join[f"Argmaxed average"].tolist(), average="macro")
    # print("="*50)
    # print([os.path.basename(os.path.dirname(fi)) for fi in files_in])
    # print(f"F1 {100*f1_all:.2f}% | Acc {100*acc_all:.2f}%")
    all_f1s += [f1_all]
    all_accs += [acc_all]
    # all_combs += [[os.path.basename(os.path.dirname(fi)) for fi in files_in]]


idx = np.argmax(all_f1s)
print(f"F1: {all_f1s[idx]} | Acc: {all_accs[idx]}")
print(all_combs[idx])

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

predictions = []
for i in range(len(df_to_join)):
    predictions += [df_to_join.loc[i][f"Argmaxed average"]]

with open(f"CairoSquad_subtask{subtask_id}_dev_{run_id}.txt", 'w') as fout:
    fout.write('\n'.join([countries[x] for x in predictions]))

with open(os.path.join(dirname, gt_name), 'r') as fin:
    countries_gt = np.array([x.strip() for x in fin.readlines()])

countries_pred = np.array([countries[x] for x in predictions])
print(np.mean(countries_gt==countries_pred))

