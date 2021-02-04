import os
import yaml
from general_utils import read_yaml_file

def save_yaml_file(content, path):
    with open(path, encoding="utf-8", mode="w") as ff:
        yaml.dump(content, ff)

command = f"python modeling.py config.yaml"
config_file_path = "./config.yaml"
config = read_yaml_file(config_file_path)

subtask = "MSA_Province"

max_seq_len = 110
config["is_province"] = True
config["is_MSA"] = True 
# config["path_to_data"] = "NADI2021_DEV.1.0/NADI2021_DEV.1.0/Subtask_1.2+2.2_DA"
config["path_to_data"] = "NADI2021_DEV.1.0/NADI2021_DEV.1.0/Subtask_1.1+2.1_MSA"

############# Run 1 #############
config["use_vert_att"] = True 
config["use_adapters"] = True 
config["max_sequence_length"] = max_seq_len
config["neptune_experiment_name"] = f"MARBERT_{subtask}_{max_seq_len}_Adapters_VAtt"
config["run_title"] = config["neptune_experiment_name"]
config["model_name_path"] = "UBC-NLP/MARBERT"
config["invsqrt-lr-scheduler"] = True

save_yaml_file(config, config_file_path)
os.system(command)

############# Run 2 #############
config["use_vert_att"] = True 
config["use_adapters"] = False 
config["max_sequence_length"] = max_seq_len
config["neptune_experiment_name"] = f"MARBERT_{subtask}_{max_seq_len}_FT_VAtt"
config["run_title"] = config["neptune_experiment_name"]
config["model_name_path"] = "UBC-NLP/MARBERT"
config["invsqrt-lr-scheduler"] = True

save_yaml_file(config, config_file_path)
os.system(command)

############# Run 3 #############
config["use_vert_att"] = False 
config["use_adapters"] = False 
config["max_sequence_length"] = max_seq_len
config["neptune_experiment_name"] = f"MARBERT_{subtask}_{max_seq_len}_FT"
config["run_title"] = config["neptune_experiment_name"]
config["model_name_path"] = "UBC-NLP/MARBERT"
config["invsqrt-lr-scheduler"] = True 

save_yaml_file(config, config_file_path)
os.system(command)

############# Run 4 #############
config["use_vert_att"] = False 
config["use_adapters"] = False 
config["max_sequence_length"] = max_seq_len
config["neptune_experiment_name"] = f"MARBERT_{subtask}_{max_seq_len}_FT_2"
config["run_title"] = config["neptune_experiment_name"]
config["model_name_path"] = "UBC-NLP/MARBERT"
config["invsqrt-lr-scheduler"] = False 

save_yaml_file(config, config_file_path)
os.system(command)
