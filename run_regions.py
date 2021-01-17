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

max_seq_len = config_file_open["max_sequence_length"]

prev_model_name = "UBC-NLP/MARBERT"
for index, region_list in enumerate(regions):
    if index in [4]: continue 
    config_file_open["neptune_experiment_name"] = f"{labels[index]}_MARBERT_Adapters_{max_seq_len}"
    config_file_open["run_title"] = f"{labels[index]}_MARBERT_Adapters_{max_seq_len}"
    config_file_open["class_index"] = index 
    config_file_open["one_class_filtration"] = region_list
    
    save_yaml_file(config_file_open, config_file_path)
    
    command = f"python modeling.py config.yaml"
    os.system(command)