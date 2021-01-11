import os
from region_country_eval import country_pred
import numpy as np
import pandas as pd 

from tqdm import tqdm
from sklearn.metrics import f1_score

countries = ['Bahrain', 'Kuwait', 'Oman', 'Qatar', 'Saudi_Arabia', 'United_Arab_Emirates', 'Yemen'] + ['Egypt', 'Sudan'] +  ['Jordan', 'Lebanon', 'Palestine', 'Syria'] + ['Algeria', 'Libya', 'Morocco', 'Tunisia'] + ['Iraq'] + ['Djibouti', 'Mauritania', 'Somalia']
countries = sorted(countries)

region_country_map = {
    # ['Bahrain', 'Kuwait', 'Oman', 'Qatar', 'Saudi_Arabia', 'United_Arab_Emirates', 'Yemen']
    0: {
        0: 1,
        1: 6,
        2: 11,
        3: 13,
        4: 14,
        5: 19,
        6: 20,
    },
    # ['Egypt', 'Sudan'] 
    1: {
        0: 3,
        1: 16
    },
    # ['Jordan', 'Lebanon', 'Palestine', 'Syria']
    2: {
        0: 5,
        1: 7,
        2: 12,
        3: 17,
    },
    # ['Algeria', 'Libya', 'Morocco', 'Tunisia']
    3: {
        0: 0, 
        1: 8,
        2: 10,
        3: 18,
    },
    # ['Iraq']
    4: {
        0: 4
    },
    # ['Djibouti', 'Mauritania', 'Somalia']
    5: {
        0: 2,
        1: 9,
        2: 15,
    }
}


if __name__ == "__main__":
    
    dirname = "checkpoints_regions"
    path = os.path.join(dirname, "RegionClassifier_3", "predictions_test.tsv")
    regions = ["Khaleegi", "Egypt_Sudan", "Levantine", "Maghrebi", "Mesopotamian", "Other"]

    region_preds = pd.read_csv(path, delimiter="\t")
    print(len(region_preds))

    countries_dfs = []
    for j, region in enumerate(regions):
        if j == 4: 
            countries_dfs += [None]
        else:
            path = os.path.join(dirname, region, f"predictions_test.tsv")
            countries_dfs += [pd.read_csv(path, delimiter="\t")]

    y_true = np.zeros(len(region_preds))
    y_pred = np.zeros(len(region_preds))
    for index, row in tqdm(region_preds.iterrows(), total=len(region_preds)):
        sid = int(row["Sentence Index"])
        gt_label = row["Labels"]
        region_pred = row["Predictions"]
        probs_regions = row["SoftMaxes"]
        probs_regions = np.array(list(map(float,probs_regions[1:-1].split())))

        scores = []

        for j, region in enumerate(regions):
            if j == 4: 
                scores += [(probs_regions[j], j, 0)]
                continue 
            
            country_preds = countries_dfs[j]
            country_preds["Sentence Index"] = pd.to_numeric(country_preds["Sentence Index"])
            country_row = country_preds[country_preds["Sentence Index"]==sid]
            probs_country = country_row["SoftMaxes"].item()
            gt_label = country_row["Labels"].item()
            probs_country = np.array(list(map(float,probs_country[1:-1].split())))
            for k in range(len(probs_country)):
                scores += [(probs_regions[j]*probs_country[k], j, k)]
        
        _, p_r, p_c = sorted(scores, reverse=True)[0]
        y_pred[index] = region_country_map[p_r][p_c]
        y_true[index] = gt_label

    acc = np.mean(y_true==y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    print(f"DEV ACC: {100*acc:.2f}%")
    print(f"MACRO F1: {100*f1:.2f}%")

    
    
