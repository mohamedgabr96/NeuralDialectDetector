import numpy as np
import json
import pandas as pd
from sklearn.metrics import f1_score

def read_from_file(file_in):
    df = pd.read_csv(file_in, delimiter="\t")
    return df

# indices_list = [60, 80, 100, 110, 120]
# indices_list = [90, 110]
# indices_list = [90, 100]
indices_list = [90, 100, 110, "100_VAT_1"]
run_id = 1

files_in = [f"/home/bkhmsi/Documents/Projects/NeuralDialectDetector/checkpoints_marbert/MARBERT_{i}/predictions_dev.tsv" for i in indices_list]

df_list = []
for i in files_in:
    df_list.append(read_from_file(i))

df_to_join = df_list[0].set_index("Sentence Index")
col_names = ["SoftMaxes"]
df_to_join[f"SoftMaxes"] = df_to_join[f"SoftMaxes"].apply(lambda x: list(map(float, x[1:-1].split())))
df_to_join[f"Argmaxed"]= df_to_join[f"SoftMaxes"].apply(lambda x: np.argmax(x))

f1_0 = f1_score(df_to_join["Labels"].tolist(), df_to_join[f"Argmaxed"].tolist(), average="macro")
df_to_join[f"isCorrect"] = df_to_join[f"Argmaxed"] == df_to_join["Labels"]
acc_0 = df_to_join[f"isCorrect"].mean()
print(f"F1 {f1_0} | Acc {acc_0}")


for index, dff in enumerate(df_list[1:]):
    df_to_join = df_to_join.join(dff.set_index("Sentence Index"), how="inner", rsuffix=str(index))
    df_to_join[f"SoftMaxes{index}"] = df_to_join[f"SoftMaxes{index}"].apply(lambda x: list(map(float,x[1:-1].split())))
    df_to_join[f"Argmaxed{index}"]= df_to_join[f"SoftMaxes{index}"].apply(lambda x: np.argmax(x))
    df_to_join[f"isCorrect{index}"] = df_to_join[f"Argmaxed{index}"] == df_to_join["Labels"]
    acc_i = df_to_join[f"isCorrect{index}"].mean()
    f1_i = f1_score(df_to_join["Labels"].tolist(), df_to_join[f"Argmaxed{index}"].tolist(), average="macro")
    print(f"F1 {f1_i} | Acc {acc_i}")

    col_names.append(f"SoftMaxes{index}")

col = df_to_join.loc[:, col_names]

df_to_join["Softmax averaged"] = col.apply(lambda x: np.mean([x[i] for i in col_names], axis=0), axis=1)
df_to_join["Argmaxed average"]= df_to_join["Softmax averaged"].apply(lambda x: np.argmax(x))
df_to_join["isCorrect average"] = df_to_join["Argmaxed average"] == df_to_join["Labels"]
acc_all = df_to_join["isCorrect average"].mean()
f1_all = f1_score(df_to_join["Labels"].tolist(), df_to_join[f"Argmaxed average"].tolist(), average="macro")
print("="*50)
print(f"F1 {100*f1_all:.2f}% | Acc {100*acc_all:.2f}%")

path = "NADI2021_DEV.1.0/NADI2021_DEV.1.0/Subtask_1.2+2.2_DA/classes_12.txt"
# path = "NADI2021_DEV.1.0/NADI2021_DEV.1.0/Subtask_1.1+2.1_MSA/classes_22.txt"
with open(path, 'r') as fin:
    countries = [x.strip() for x in fin.readlines()]
countries.sort()

predictions = []
for i in range(5000):
    predictions += [df_to_join.loc[i][f"Argmaxed average"]]

with open(f"CairoSquad_subtask12_dev_{run_id}.txt", 'w') as fout:
    fout.write('\n'.join([countries[x] for x in predictions]))

path = "NADI2021_DEV.1.0/NADI2021_DEV.1.0/Subtask_1.2+2.2_DA/subtask12_GOLD.txt"
# path = "NADI2021_DEV.1.0/NADI2021_DEV.1.0/Subtask_1.1+2.1_MSA/subtask21_GOLD.txt"
with open(path, 'r') as fin:
    countries_gt = np.array([x.strip() for x in fin.readlines()])

countries_pred = np.array([countries[x] for x in predictions])
print(np.mean(countries_gt==countries_pred))

