import os
import yaml
from general_utils import read_yaml_file

def save_yaml_file(content, path):
    with open(path, encoding="utf-8", mode="w") as ff:
        yaml.dump(content, ff)

regions = [
    ["Kuwait", "Oman", "United_Arab_Emirates", "Saudi_Arabia", "Bahrain", "Yemen", "Qatar"],
    ["Egypt", "Sudan"],
    ["Lebanon", "Jordan", "Palestine", "Syria"],
    ["Tunisia", "Libya", "Algeria", "Morocco"],
    ["Iraq"],
    ["Somalia", "Djibouti", "Mauritania"]
]

labels = [
    "Khaleegi",
    "Egypt_Sudan",
    "Levantine",
    "Maghrebi",
    "Iraq",
    "Other"
]

config_file_path = "./config.yaml"
config_file_open = read_yaml_file(config_file_path)

for index, region in enumerate(labels):
    if index in [4]: continue 
    config_file_open["use_neptune"] = False
    config_file_open["run_title"] = f"{region}_MARBERT_Adapters_60"
    config_file_open["class_index"] = -1 
    config_file_open["one_class_filtration"] = None 
    config_file_open["model_name_path"] = os.path.join("checkpoints_marbert", f"{region}_MARBERT_Adapters_60")
    config_file_open["num_epochs"] = 0
    config_file_open["num_labels"] = len(regions[index])
    config_file_open["use_regional_mapping"] = False

    save_yaml_file(config_file_open, config_file_path)
    
    command = f"python modeling.py config.yaml"
    os.system(command)