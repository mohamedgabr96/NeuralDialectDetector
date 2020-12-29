from transformers import AutoTokenizer, AutoModel, AutoConfig, AdamW, get_linear_schedule_with_warmup
from dataset_utils import parse_and_generate_loaders
from general_utils import read_yaml_file, save_model, evaluate_predictions, update_dict_of_agg, save_json
import logging
import torch
from tqdm import tqdm, trange
import numpy as np
import os
import neptune
import random 
import model as model_classes

logger = logging.getLogger(__name__)


class Trainer():
    def __init__(self, config_file_path="config.yaml"):
        self.configs = read_yaml_file(config_file_path)
        self.model_name_path = self.configs["model_name_path"]
        logging.basicConfig(filename=os.path.join(self.configs["checkpointing_path"], 'training_log.log'), level=logging.DEBUG)
        neptune.init(project_qualified_name='mohamedgabr96/sandbox',
             api_token=self.configs["neptuneaiAPI"],
             )

    @staticmethod
    def set_seeds(seed_val):
        np.random.seed(seed_val)
        random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def log_single_metrics_to_neptune(self):
        neptune.log_metric('batch_size', self.configs["batch_size"])
        neptune.log_metric('classif_dropout_rate', self.configs["classif_dropout_rate"])
        neptune.log_metric('initial_learning_rate', self.configs["initial_learning_rate"])
        neptune.log_metric('adam_epsilon', self.configs["adam_epsilon"])
        neptune.log_metric('warmup_steps', self.configs["warmup_steps"])
        neptune.log_metric('num_epochs', self.configs["num_epochs"])
        neptune.log_metric('masking_percentage', self.configs["masking_percentage"])

    def train(self):
        neptune.create_experiment(name=self.configs["neptune_experiment_name"])
        self.log_single_metrics_to_neptune()
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_path)
        model_config = AutoConfig.from_pretrained(self.model_name_path)

        # Generate Loaders
        train_loader, dev_loader, test_loader, no_labels = parse_and_generate_loaders(self.configs["path_to_data"], tokenizer, batch_size=self.configs["batch_size"], masking_percentage=self.configs["masking_percentage"], class_to_filter=self.configs["one_class_filtration"])
        self.configs["num_labels"] = no_labels

        # model = AutoModel.from_pretrained(self.model_name_path)
        # Instantiate Model
        model = getattr(model_classes, self.configs["model_class"]).from_pretrained(self.model_name_path,
                                                            config=model_config,
                                                            args=self.configs)
        model.to(self.configs["device"])
        total_steps = len(train_loader) * self.configs["num_epochs"]

        # Initialize Optimizers
        if self.configs["model_class"] == "ArabicDialectBERTMaskedLM":
            optimizer = AdamW(model.parameters(), lr=self.configs["initial_learning_rate"], eps=self.configs["adam_epsilon"])
        else:
            optimizer = AdamW([
                {
                    'params': model.bert.parameters()
                },
                {
                    'params': model.classif_head.parameters(),
                    'lr': 1.e-3
                }
            ], lr=self.configs["initial_learning_rate"], eps=self.configs["adam_epsilon"])
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.configs["warmup_steps"], num_training_steps=total_steps)

        model.zero_grad()

        # Training Loop
        best_model_path = ""
        last_model_path = ""
        no_epochs = trange(self.configs["num_epochs"], desc="Epoch Number")
        global_step = 0
        training_loss = 0.0
        best_dev_loss = np.inf
        curr_dev_loss = np.inf
        early_stop_count_patience = 0
        to_early_stop = False
        for _ in no_epochs:
            no_batches = tqdm(train_loader, desc="Batches Loop")
            for batch in no_batches:
                         
                model.train()
                
                batch = [x.to(self.configs["device"]) for x in batch]
                outputs = model(input_ids=batch[0], attention_mask=batch[1], token_type_ids=batch[2], class_label_ids=batch[3], input_ids_masked=batch[4])
                loss = outputs[0]

                loss.backward()

                training_loss += loss.item()

                neptune.log_metric('train_loss', x=global_step, y=loss.item())
                neptune.log_metric('learning_rate_body', x=global_step, y=optimizer.param_groups[0]['lr'])
                if len(optimizer.param_groups) > 1:
                    neptune.log_metric('learning_rate_head', x=global_step, y=optimizer.param_groups[1]['lr'])

                optimizer.step()
                scheduler.step()  
                model.zero_grad()
                global_step += 1

                if global_step % self.configs["improvement_check_freq"] == 0:
                    dev_accuracy, curr_dev_loss = evaluate_predictions(model, dev_loader, self.configs["model_class"], device=self.configs["device"])
                    neptune.log_metric('dev_loss', x=global_step, y=curr_dev_loss)
                    neptune.log_metric('dev_accuracy', x=global_step, y=dev_accuracy)
                    early_stop_count_patience += 1

                if self.configs["early_stopping"] and early_stop_count_patience > self.configs["early_stopping_patience"]:
                    logger.info(f"Early Stopping, no improvements on dev score for no patience steps")
                    to_early_stop = True
                    break

                if self.configs["checkpoint_on_improvement"] and curr_dev_loss < best_dev_loss:
                    logger.info(f"Dev Loss Reduction from {best_dev_loss} to {curr_dev_loss}")
                    best_model_path = save_model(model, tokenizer, self.configs["checkpointing_path"], self.configs, step_no=global_step, current_dev_score=best_dev_loss)
                    best_dev_loss = curr_dev_loss
                    early_stop_count_patience = 0

                if self.configs["checkpointing_on"] and global_step % self.configs["checkpointing_freq"] == 0:
                    logger.info(f"Checkpointing...")
                    save_model(model, tokenizer, self.configs["checkpointing_path"], self.configs, step_no=global_step, current_dev_score=best_dev_loss)
            if to_early_stop:
                break
       
        loss_final = training_loss / global_step

        # Final Evaluation Loop

        final_dev_accuracy, final_dev_loss = evaluate_predictions(model, dev_loader, self.configs["model_class"], device=self.configs["device"])
        final_test_accuracy, final_test_loss = evaluate_predictions(model, test_loader, self.configs["model_class"], device=self.configs["device"])

        neptune.log_metric('dev_loss', x=global_step, y=final_dev_loss)
        neptune.log_metric('dev_accuracy', x=global_step, y=final_dev_accuracy)

        # Final Model Saving
        if self.configs["save_final_model"]:
            last_model_path = save_model(model, tokenizer, self.configs["checkpointing_path"], self.configs, step_no=global_step, current_dev_score=final_dev_accuracy)

        logger.info(f"Finished Training and Evaluation, final dev accuracy is {final_dev_accuracy}")

        model_path_to_return = best_model_path if len(best_model_path) > 0 else last_model_path

        return final_dev_accuracy, model_path_to_return

    def train_with_multiple_seeds(self, no_times, seeds_from_config=False):
        seed_list = self.configs["seed_list"]
        logger.info(f"Training with {no_times} seeds..")
        dev_accuracy_list = []
        model_pathes = []
        for i in range(no_times):
            logger.info(f"Training with the seed no {i} started..")
            if seeds_from_config:
                self.set_seeds(seed_list[i])
            temp_accuracy, best_model_path = self.train()
            dev_accuracy_list.append(temp_accuracy)
            model_pathes.append(best_model_path)
            logger.info(f"Training with the seed no {i} ended..")

        final_aggregation = np.mean(dev_accuracy_list)
        logger.info(f"Training with multiple seeds ended.. Final aggregation of dev scores {final_aggregation}")
        return np.mean(dev_accuracy_list), model_pathes

    def evaluate_from_path(self, model_path, evaluate_on_train=True):
        dict_of_results = {}

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model_config = AutoConfig.from_pretrained(model_path)

        # Generate Loaders
        train_loader, dev_loader, test_loader, no_labels = parse_and_generate_loaders(self.configs["path_to_data"], tokenizer, batch_size=self.configs["batch_size"], masking_percentage=self.configs["masking_percentage"], class_to_filter=self.configs["one_class_filtration"])
        self.configs["num_labels"] = no_labels

        # Instantiate Model
        model = getattr(model_classes, self.configs["model_class"]).from_pretrained(model_path,
                                                            config=model_config,
                                                            args=self.configs)

        model.to(self.configs["device"])

        final_dev_accuracy, final_dev_loss = evaluate_predictions(model, dev_loader, self.configs["model_class"], device=self.configs["device"])
        final_test_accuracy, final_test_loss = evaluate_predictions(model, test_loader, self.configs["model_class"], device=self.configs["device"])

        dict_of_results["DEV"] = {"Accuracy": final_dev_accuracy, "Loss": final_dev_loss} 
        dict_of_results["TEST"] = {"Accuracy": final_test_accuracy, "Loss": final_test_loss}

        if evaluate_on_train:
            final_train_accuracy, final_train_loss = evaluate_predictions(model, train_loader, self.configs["model_class"], device=self.configs["device"])
            dict_of_results["TRAIN"] = {"Accuracy": final_train_accuracy, "Loss": final_train_loss}

        return dict_of_results

    def train_and_evaluate_with_multiple_seeds(self, no_times, seeds_from_config=False):
        accuracy_agg, model_pathes = self.train_with_multiple_seeds(no_times, seeds_from_config=seeds_from_config)

        dict_of_seed_results = {}
        aggregation_dict = {"TRAIN": {"Accuracy": [], "Loss": []}, "DEV": {"Accuracy": [], "Loss": []}, "TEST": {"Accuracy": [], "Loss": []}}
        for index, path in enumerate(model_pathes):
            dict_of_results = self.evaluate_from_path(path)
            dict_of_seed_results[f"seed_{index}"] = dict_of_results
            aggregation_dict = update_dict_of_agg(aggregation_dict, dict_of_results)

        # Add a Field of Aggregation
        for split in ["TRAIN", "DEV", "TEST"]:
            aggregation_dict[split]["Accuracy"] = np.mean(aggregation_dict[split]["Accuracy"])
            aggregation_dict[split]["Loss"] = np.mean(aggregation_dict[split]["Loss"])

        dict_of_seed_results["Agg"] = aggregation_dict

        save_json(os.path.join(self.configs["checkpointing_path"], "final_scores.json"), dict_of_seed_results)
        neptune.stop()
        return dict_of_seed_results


if __name__ == "__main__":
    trainer_class = Trainer()
    # trainer_class.train()
    # trainer_class.train_with_multiple_seeds(3)
    trainer_class.train_and_evaluate_with_multiple_seeds(1, seeds_from_config=True)
