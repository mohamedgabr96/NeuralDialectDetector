import os
import yaml
from general_utils import read_yaml_file

def save_yaml_file(content, path):
    with open(path, encoding="utf-8", mode="w") as ff:
        yaml.dump(content, ff)


config_file_path = "./config.yaml"
config_file_open = read_yaml_file(config_file_path)

for max_seq_len in [110]:

    config_file_open["max_sequence_length"] = max_seq_len
    config_file_open["neptune_experiment_name"] = f"MARBERT_MSA_Province_{max_seq_len}"
    config_file_open["run_title"] = f"MARBERT_MSA_Province_{max_seq_len}"
    
    save_yaml_file(config_file_open, config_file_path)
    
    command = f"python modeling.py config.yaml"
    os.system(command)