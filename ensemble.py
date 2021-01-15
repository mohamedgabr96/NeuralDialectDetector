import numpy as np
import json
import pandas as pd
from sklearn.metrics import f1_score

def read_from_file(file_in):
    df = pd.read_csv(file_in, delimiter="\t")
    return df

indices_list = [50, 80, 120]

files_in = [f"/home/bkhmsi/Documents/Projects/NeuralDialectDetector/checkpoints_marbert/MARBERT_{i}/predictions_dev.tsv" for i in indices_list]

df_list = []
for i in files_in:
    df_list.append(read_from_file(i))

df_to_join = df_list[0].set_index("Sentence Index")
col_names = ["SoftMaxes"]
df_to_join[f"SoftMaxes"] = df_to_join[f"SoftMaxes"].apply(lambda x: list(map(float, x[1:-1].split())))
df_to_join[f"Argmaxed"]= df_to_join[f"SoftMaxes"].apply(lambda x: np.argmax(x))

f1_0 = f1_score(df_to_join["Labels"].tolist(), df_to_join[f"Argmaxed"].tolist(), average="macro")
print(f1_0)

df_to_join[f"isCorrect"] = df_to_join[f"Argmaxed"] == df_to_join["Labels"]
# print(df_to_join[f"isCorrect"].mean())

for index, dff in enumerate(df_list[1:]):
    df_to_join = df_to_join.join(dff.set_index("Sentence Index"), how="inner", rsuffix=str(index))
    df_to_join[f"SoftMaxes{index}"] = df_to_join[f"SoftMaxes{index}"].apply(lambda x: list(map(float,x[1:-1].split())))
    df_to_join[f"Argmaxed{index}"]= df_to_join[f"SoftMaxes{index}"].apply(lambda x: np.argmax(x))
    df_to_join[f"isCorrect{index}"] = df_to_join[f"Argmaxed{index}"] == df_to_join["Labels"]
    # print(df_to_join[f"isCorrect{index}"].mean())
    f1_0 = f1_score(df_to_join["Labels"].tolist(), df_to_join[f"Argmaxed{index}"].tolist(), average="macro")
    print(f1_0)

    col_names.append(f"SoftMaxes{index}")

col = df_to_join.loc[:, col_names]

df_to_join["Softmax averaged"] = col.apply(lambda x: np.mean([x[i] for i in col_names], axis=0), axis=1)
df_to_join["Argmaxed average"]= df_to_join["Softmax averaged"].apply(lambda x: np.argmax(x))
df_to_join["isCorrect average"] = df_to_join["Argmaxed average"] == df_to_join["Labels"]
accuracy = df_to_join["isCorrect average"].mean()
f1_0 = f1_score(df_to_join["Labels"].tolist(), df_to_join[f"Argmaxed average"].tolist(), average="macro")
print(f1_0)
print("Accuracy")
# print(accuracy)
print("Done")