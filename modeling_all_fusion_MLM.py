import os
from general_utils import read_yaml_file
import yaml

def save_yaml_file(content, path):
    with open(path, encoding="utf-8", mode="w") as ff:
        yaml.dump(content, ff)

configs_folder = "fusion_config_folder"
file_path_classes = "dataset_dummy/classes.txt"

all_configs_list = os.listdir(configs_folder)

config_file_path = os.path.join(configs_folder, all_configs_list[0])

config_file_open = read_yaml_file(config_file_path)

countries_list = open(file_path_classes, encoding="utf-8").readlines()
countries_list = [x.strip("\n") for x in countries_list]

prev_model_name = "checkpoints/adapter_bert_arabic_w_pretrainedfusionstuff-20210105T124954Z-001/adapter_bert_arabic_w_pretrainedfusionstuff"
for index, country in enumerate(countries_list):
    config_file_open["one_class_filtration"] = country
    config_file_open["current_adapter_to_train"] = index
    config_file_open["neptune_experiment_name"] = f"adapter_fusion_stage_1_{country}"
    config_file_open["model_name_path"] = prev_model_name
    config_file_open["run_title"] = f"adapter_fusion_{country}"
    prev_model_name = os.path.join(config_file_open["checkpointing_path"], f"adapter_fusion_{country}")

    ## Save new config
    save_yaml_file(config_file_open, config_file_path)

    ## Run
    from modeling import Trainer
    temp_trainer_class = Trainer(config_file_path=config_file_path)
    temp_trainer_class.train_and_evaluate_with_multiple_seeds(1, seeds_from_config=True)
    del temp_trainer_class
