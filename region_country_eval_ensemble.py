import os
import numpy as np
import pandas as pd 

from tqdm import tqdm
from sklearn.metrics import f1_score

if __name__ == "__main__":

    dirname = "checkpoints_marbert"
    path_1 = os.path.join(dirname, "RegionClassifier_100", "predictions_test_final.tsv")
    path_2 = os.path.join(dirname, "RegionClassifier_110", "predictions_test_final.tsv")
    path_3 = os.path.join(dirname, "RegionClassifier_60", "predictions_test_final.tsv")

    df_1 = pd.read_csv(path_1, delimiter="\t")
    df_2 = pd.read_csv(path_2, delimiter="\t")
    df_3 = pd.read_csv(path_3, delimiter="\t")

    y_true, y_pred = np.zeros(len(df_1)), np.zeros(len(df_1))
    for index, row in tqdm(df_1.iterrows(), total=len(df_1)):
        sid = row["Sentence Index"]
        gt_label = row["Ground Truth"]
        
        probs_1 = row["FinalSoftmax"]
        probs_1 = np.array(list(map(float,probs_1[1:-1].split(','))))
        
        row_2 = df_2[df_2["Sentence Index"]==sid]
        probs_2 = row_2["FinalSoftmax"].item()
        probs_2 = np.array(list(map(float, probs_2[1:-1].split(','))))

        row_3 = df_3[df_3["Sentence Index"]==sid]
        probs_3 = row_3["FinalSoftmax"].item()
        probs_3 = np.array(list(map(float, probs_3[1:-1].split(','))))

        final_probs = probs_1*probs_2*probs_3

        y_pred[index] = np.argmax(final_probs)
        y_true[index] = int(gt_label)

    acc = np.mean(y_true==y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    print(f"DEV ACC: {100*acc:.2f}%")
    print(f"MACRO F1: {100*f1:.2f}%")