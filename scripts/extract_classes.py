import os 
import numpy as np
import pandas as pd 

if __name__ == "__main__":

    # path = "NADI2021_DEV.1.0/NADI2021_DEV.1.0/Subtask_1.2+2.2_DA/DA_dev_labeled.tsv"
    path = "NADI2021_DEV.1.0/NADI2021_DEV.1.0/Subtask_1.1+2.1_MSA/MSA_dev_labeled.tsv"
    df = pd.read_csv(path, delimiter='\t')
    
    labels_1 = sorted(df["#3_country_label"].unique().tolist())
    labels_2 = sorted(df["#4_province_label"].unique().tolist())

    print(f"n1={len(labels_1)} | n2={len(labels_2)}")

    path = os.path.join(os.path.dirname(path), "classes_22.txt")
    with open(path, 'w') as fout:
        fout.write('\n'.join(labels_1))
    
    path = os.path.join(os.path.dirname(path), "classes_22.txt")
    with open(path, 'w') as fout:
        fout.write('\n'.join(labels_2))
