import os
import yaml
from general_utils import read_yaml_file

def save_yaml_file(content, path):
    with open(path, encoding="utf-8", mode="w") as ff:
        yaml.dump(content, ff)

command = f"python modeling.py config.yaml"
config_file_path = "./config.yaml"
config = read_yaml_file(config_file_path)

subtask = "MSA_Country"

config["is_province"] = False
config["is_MSA"] = True 
# config["path_to_data"] = "NADI2021_DEV.1.0/NADI2021_DEV.1.0/Subtask_1.2+2.2_DA"
config["path_to_data"] = "NADI2021_DEV.1.0/NADI2021_DEV.1.0/Subtask_1.1+2.1_MSA"

############# Run 1 #############
max_seq_len = 60
config["use_vert_att"] = False 
config["max_sequence_length"] = max_seq_len
config["neptune_experiment_name"] = f"MARBERT_{subtask}_{max_seq_len}"
config["run_title"] = f"MARBERT_{subtask}_{max_seq_len}"
config["model_name_path"] = os.path.join(config["checkpointing_path"], config["run_title"])

save_yaml_file(config, config_file_path)
os.system(command)

############# Run 2 #############
max_seq_len = 90
config["max_sequence_length"] = max_seq_len
config["neptune_experiment_name"] = f"MARBERT_{subtask}_{max_seq_len}_VAT"
config["run_title"] = f"MARBERT_{subtask}_{max_seq_len}_VAT"
config["use_vert_att"] = True
config["vatt-positional-keys"] = "random"
config["model_name_path"] = os.path.join(config["checkpointing_path"], config["run_title"])

save_yaml_file(config, config_file_path)
os.system(command)

############# Run 3 #############
config["neptune_experiment_name"] = f"MARBERT_{subtask}_{max_seq_len}_VAT_1"
config["run_title"] = f"MARBERT_{subtask}_{max_seq_len}_VAT_1"
config["vatt-positional-keys"] = "sinosoid"
config["model_name_path"] = os.path.join(config["checkpointing_path"], config["run_title"])

# save_yaml_file(config, config_file_path)
# os.system(command)