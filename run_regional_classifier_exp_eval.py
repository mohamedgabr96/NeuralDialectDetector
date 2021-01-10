import numpy as np
import os
from dataset_utils import parse_mapping_list
from sklearn.metrics import f1_score

## This script assumes that evaluation is run, thus the predictions per sentence are dumped
# ath_to_folder_data = ""
dirname = "./checkpoints_regions"
region_to_model_pathes = ["Levantine", "Egypt_Sudan", "Khaleegi", "Maghrebi", "Other"]

region_to_country_mapping = {}

# mapping_list = parse_mapping_list(path_to_folder_data)

def read_predictions(model_path, split):
    path_to_file = os.path.join(model_path, f"predictions_{split}.tsv")
    with open(path_to_file, encoding="utf-8") as ff:
        file_opened = ff.readlines()

    file_opened = [x.strip().split("\t") for x in file_opened]

    return file_opened[2:]

def calculate_accuracy(list_of_predictions):
    corrects = [int(x[1] == x[2]) for x in list_of_predictions]
    acc = corrects.count(1) / len(list_of_predictions)
    return acc
 
all_preds = []
y_preds = []
y_labels = []
for path in region_to_model_pathes:
    path = os.path.join(dirname, path)
    all_preds += read_predictions(path, "test")

    y_preds += [x[1] for x in all_preds]
    y_labels += [x[2] for x in all_preds]


f1 = f1_score(y_labels, y_preds, average='macro')
accuracy = calculate_accuracy(all_preds)

print()


## TO: DO: 

# Map regions from ground truth for labels

# Map regions for predictions by:
  # - getting the list of country names for each regions then map
  # - Levantine: 

