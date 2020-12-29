import os

configs_folder = "config_folder_trial"

all_configs_list = os.listdir(configs_folder)

for config in all_configs_list:
    from modeling import Trainer
    temp_trainer_class = Trainer(config_file_path=os.path.join(configs_folder, config))
    temp_trainer_class.train_and_evaluate_with_multiple_seeds(1, seeds_from_config=True)
    del temp_trainer_class
