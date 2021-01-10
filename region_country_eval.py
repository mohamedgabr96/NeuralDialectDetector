import os
import numpy as np
import pandas as pd 

from tqdm import tqdm
from sklearn.metrics import f1_score

region_country_map = {
    # ['Bahrain', 'Kuwait', 'Oman', 'Qatar', 'Saudi_Arabia', 'United_Arab_Emirates', 'Yemen']
    0: {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
    },
    # ['Egypt', 'Sudan'] 
    1: {
        0: 7,
        1: 8
    },
    # ['Jordan', 'Lebanon', 'Palestine', 'Syria']
    2: {
        0: 9,
        1: 10,
        2: 11,
        3: 12,
    },
    # ['Algeria', 'Libya', 'Morocco', 'Tunisia']
    3: {
        0: 13, 
        1: 14,
        2: 15,
        3: 16,
    },
    # ['Iraq']
    4: {
        0: 17
    },
    # ['Djibouti', 'Mauritania', 'Somalia']
    5: {
        0: 18,
        1: 19,
        2: 20,
    }
}

def country_pred(region_id, sid, dtype):    

    if region_id in [4]:
        return region_country_map[region_id][0]    

    path = os.path.join(dirname, regions[region_id], f"predictions_{dtype}.tsv")
    country_preds = pd.read_csv(path, delimiter="\t")

    column = "Predictions" if dtype == "test" else "Labels"
    local_pred = country_preds[column][country_preds["Sentence Index"]==sid].item()

    return region_country_map[region_id][local_pred]

if __name__ == "__main__":
    
    dirname = "checkpoints_regions"
    path = os.path.join(dirname, "RegionClassifier", "predictions_test.tsv")
    regions = ["Khaleegi", "Egypt_Sudan", "Levantine", "Maghrebi", "Mesopotamian", "Other"]

    region_preds = pd.read_csv(path, delimiter="\t")

    y_true = np.zeros(len(region_preds))
    y_pred = np.zeros(len(region_preds))
    for index, row in tqdm(region_preds.iterrows(), total=len(region_preds)):
        sid = row["Sentence Index"]
        gt_label = row["Labels"]
        region_pred = row["Predictions"]

        y_pred[index] = country_pred(region_pred, sid, "test")
        y_true[index] = country_pred(gt_label, sid, "dev") 

    acc = np.mean(y_true==y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    print(f"DEV ACC: {100*acc:.2f}%")
    print(f"MACRO F1: {100*f1:.2f}%")

    
    
