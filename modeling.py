from sklearn.utils.validation import check_is_fitted
from transformers import AutoTokenizer, AutoModel, AutoConfig, AdamW, get_linear_schedule_with_warmup
from dataset_utils import parse_and_generate_loaders
from general_utils import read_yaml_file, save_model, evaluate_predictions, update_dict_of_agg, save_json, dump_predictions
import logging
import torch
from tqdm import tqdm, trange
import numpy as np
import os
import neptune
import random 
import model as model_classes
from torch.optim.lr_scheduler import CyclicLR, LambdaLR

logger = logging.getLogger(__name__)

global_step = 0

class InvSqrtLR(LambdaLR):
    def __init__(
            self, optim,
            num_warmup: int, max_factor: float = 10, min_factor: float = 0.01,
            mini_epoch_size=1,
    ):
        self.num_warmup = num_warmup
        self.max_factor = max_factor
        self.min_factor = min_factor
        self.mini_epoch_sz = mini_epoch_size
        super().__init__(optim, self.lr_lambda)

        logger.info(f'InvSqrtLR dict: {self.__dict__}')

    def lr_lambda(self, iteration: int) -> float:
        iteration = iteration // self.mini_epoch_sz
        
        if iteration < self.num_warmup:
            step = (self.max_factor - self.min_factor) / float(self.num_warmup)
            fac  = iteration * step
        else:
            fac = self.max_factor / np.sqrt(iteration)
            fac = max(fac, self.min_factor)
        neptune.log_metric('InvSqrtLR_factor', x=global_step, y=fac)
        return fac

class Trainer():
    def __init__(self, config_file_path="config.yaml"):
        self.configs = read_yaml_file(config_file_path)
        self.model_name_path = self.configs["model_name_path"]
        logging.basicConfig(filename=os.path.join(self.configs["checkpointing_path"], 'training_log.log'), level=logging.DEBUG)
        if self.configs["use_neptune"]:
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
        if self.configs["use_neptune"]:
            neptune.create_experiment(name=self.configs["neptune_experiment_name"])
            self.log_single_metrics_to_neptune()
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_path)
        model_config = AutoConfig.from_pretrained(self.model_name_path)

        # Generate Loaders
        train_loader, dev_loader, test_loader, no_labels, cls_weights = parse_and_generate_loaders(self.configs["path_to_data"], tokenizer, batch_size=self.configs["batch_size"], masking_percentage=self.configs["masking_percentage"], class_to_filter=self.configs["one_class_filtration"], filter_w_indexes=self.configs["indexes_filtration_path"], pred_class=self.configs["class_index"], use_regional_mapping=self.configs["use_regional_mapping"], max_seq_len=self.configs["max_sequence_length"], balance_data_max_examples=self.configs["max_ex_per_class"], is_province=self.configs["is_province"], is_MSA=self.configs["is_MSA"], sampler_imbalance=self.configs["handle_imbalance_sampler"])
        self.configs["num_labels"] = self.configs.get("num_labels", no_labels)
        self.configs["cls_weights"] = cls_weights

        # model = AutoModel.from_pretrained(self.model_name_path)
        # Instantiate Model
        self.configs["mask_id"] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        model = getattr(model_classes, self.configs["model_class"]).from_pretrained(self.model_name_path,
                                                            config=model_config,
                                                            args=self.configs)
        model.to(self.configs["device"])
        total_steps = len(train_loader) * self.configs["num_epochs"]

        # Initialize Optimizers
        if self.configs["model_class"] == "ArabicDialectBERTMaskedLM":
            model.init_cls_weights()
            optimizer = AdamW(model.parameters(), lr=self.configs["initial_learning_rate"], eps=self.configs["adam_epsilon"])
        else:
            optimizer = AdamW([
                {
                    'params': model.bert.parameters()
                },
                {
                    'params': model.classif_head.parameters(),
                    'lr': 1.e-3 #5.e-5
                }
            ], lr=self.configs["initial_learning_rate"], eps=self.configs["adam_epsilon"])
        mini_epoch_size = self.configs.get('lr-mini-epoch-size', 1)
        scheduler = InvSqrtLR(optimizer,
            num_warmup=self.configs["warmup_steps"] // mini_epoch_size,
            mini_epoch_size=mini_epoch_size,
        )
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.configs["warmup_steps"], num_training_steps=total_steps)
        # scheduler = CyclicLR(optimizer, base_lr=5.e-6, max_lr=5.e-5, step_size_up=657, cycle_momentum=False)

        model.zero_grad()

        # Training Loop
        best_model_path = ""
        last_model_path = ""
        no_epochs = trange(self.configs["num_epochs"], desc="Epoch Number")

        #assert no_labels == self.configs["num_labels"] or no_epochs==0, "Specified Number of Labels Not Equal to Labels in Model"

        global global_step
        global_step = 0
        training_loss = 0.0
        best_dev_loss = np.inf
        curr_dev_loss = np.inf
        best_dev_f1 = 0
        curr_dev_f1 = 0
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
                if self.configs["use_neptune"]:
                    neptune.log_metric('train_loss', x=global_step, y=loss.item())
                    neptune.log_metric('learning_rate_body', x=global_step, y=optimizer.param_groups[0]['lr'])
                    if len(optimizer.param_groups) > 1:
                        neptune.log_metric('learning_rate_head', x=global_step, y=optimizer.param_groups[1]['lr'])

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if global_step % self.configs["improvement_check_freq"] == 0:
                    curr_dev_f1, dev_accuracy, curr_dev_loss = evaluate_predictions(model, dev_loader, self.configs["model_class"], device=self.configs["device"])
                    if self.configs["use_neptune"]:
                        neptune.log_metric('dev_loss', x=global_step, y=curr_dev_loss)
                        neptune.log_metric('dev_accuracy', x=global_step, y=dev_accuracy)
                        neptune.log_metric('dev_f1', x=global_step, y=curr_dev_f1)
                    early_stop_count_patience += 1

                if self.configs["early_stopping"] and early_stop_count_patience > self.configs["early_stopping_patience"]:
                    logger.info(f"Early Stopping, no improvements on dev score for no patience steps")
                    to_early_stop = True
                    break

                if self.configs["checkpoint_on_improvement"] and curr_dev_f1 > best_dev_f1:
                    logger.info(f"Dev Loss Reduction from {best_dev_loss} to {curr_dev_loss}")
                    best_model_path = save_model(model, tokenizer, self.configs["checkpointing_path"], self.configs, step_no=global_step, current_dev_score=curr_dev_f1)
                    best_dev_f1 = curr_dev_f1
                    best_dev_loss = curr_dev_loss
                    early_stop_count_patience = 0

                if self.configs["checkpointing_on"] and global_step % self.configs["checkpointing_freq"] == 0:
                    logger.info(f"Checkpointing...")
                    save_model(model, tokenizer, self.configs["checkpointing_path"], self.configs, step_no=global_step, current_dev_score=best_dev_loss)
            if to_early_stop:
                break
       
        loss_final = (training_loss / global_step) if global_step > 0 else 0

        # Final Evaluation Loop
        if self.configs["num_epochs"] > 0:
            final_dev_f1, final_dev_accuracy, final_dev_loss = evaluate_predictions(model, dev_loader, self.configs["model_class"], device=self.configs["device"])
            final_test_f1, final_test_accuracy, final_test_loss = evaluate_predictions(model, test_loader, self.configs["model_class"], device=self.configs["device"], isTest=True)
        else:
            final_dev_accuracy = 0

        if self.configs["use_neptune"]:
            neptune.log_metric('dev_loss', x=global_step, y=final_dev_loss)
            neptune.log_metric('dev_accuracy', x=global_step, y=final_dev_accuracy)
            neptune.log_metric('dev_f1', x=global_step, y=final_dev_f1)
            
        isTest_flag_for_dev_train = not (no_labels == self.configs["num_labels"])

        if self.configs["num_epochs"] > 0:
            final_dev_f1, final_dev_accuracy, final_dev_loss = evaluate_predictions(model, dev_loader, self.configs["model_class"], device=self.configs["device"], isTest=isTest_flag_for_dev_train)
            final_test_f1, final_test_accuracy, final_test_loss = evaluate_predictions(model, test_loader, self.configs["model_class"], device=self.configs["device"], isTest=True)

   
        # Final Model Saving
        if self.configs["save_final_model"]:
            last_model_path = save_model(model, tokenizer, self.configs["checkpointing_path"], self.configs, step_no=global_step, current_dev_score=final_dev_accuracy)
        else:
            last_model_path = self.model_name_path 
            
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

        dev_accuracy_list = np.array(dev_accuracy_list)
        final_aggregation = np.mean(dev_accuracy_list)
        logger.info(f"Training with multiple seeds ended.. Final aggregation of dev scores {final_aggregation}")
        return np.mean(dev_accuracy_list), model_pathes

    def evaluate_from_path(self, model_path, evaluate_on_train=False):
        dict_of_results = {}

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model_config = AutoConfig.from_pretrained(model_path)

        # Generate Loaders
        train_loader, dev_loader, test_loader, no_labels, _ = parse_and_generate_loaders(self.configs["path_to_data"], tokenizer, batch_size=self.configs["batch_size"], masking_percentage=self.configs["masking_percentage"], class_to_filter=self.configs["one_class_filtration"], filter_w_indexes=self.configs["indexes_filtration_path"], pred_class=self.configs["class_index"], use_regional_mapping=self.configs["use_regional_mapping"], max_seq_len=self.configs["max_sequence_length"], balance_data_max_examples=self.configs["max_ex_per_class"], is_province=self.configs["is_province"], is_MSA=self.configs["is_MSA"], sampler_imbalance=self.configs["handle_imbalance_sampler"])
        self.configs["num_labels"] = self.configs.get("num_labels", no_labels)

        isTest_flag_for_dev_train = not (no_labels == self.configs["num_labels"])

        # Instantiate Model
        self.configs["mask_id"] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        model = getattr(model_classes, self.configs["model_class"]).from_pretrained(model_path,
                                                            config=model_config,
                                                            args=self.configs)

        model.to(self.configs["device"])

        final_dev_f1, final_dev_accuracy, final_dev_loss, y_true_dev, y_pred_dev, sentence_id_dev, logits_list_dev = evaluate_predictions(model, dev_loader, self.configs["model_class"], device=self.configs["device"], return_pred_lists=True, isTest=isTest_flag_for_dev_train)
        dump_predictions(sentence_id_dev, logits_list_dev, y_pred_dev, y_true_dev, os.path.join(model_path, "predictions_dev.tsv"))
        
        final_test_f1, final_test_accuracy, final_test_loss, y_true_test, y_pred_test, sentence_id_test, logits_list_test = evaluate_predictions(model, test_loader, self.configs["model_class"], device=self.configs["device"], return_pred_lists=True, isTest=True)
        dump_predictions(sentence_id_test, logits_list_test, y_pred_test, y_true_test, os.path.join(model_path, "predictions_test.tsv"))

        dict_of_results["DEV"] = {"F1": final_dev_f1, "Accuracy": final_dev_accuracy, "Loss": final_dev_loss} 
        dict_of_results["TEST"] = {"F1": final_test_f1, "Accuracy": final_test_accuracy, "Loss": final_test_loss}

        if evaluate_on_train:
            final_train_f1, final_train_accuracy, final_train_loss, y_true_train, y_pred_train, sentence_id_train, logits_list_train = evaluate_predictions(model, train_loader, self.configs["model_class"], device=self.configs["device"], return_pred_lists=True, isTest=isTest_flag_for_dev_train)
            dump_predictions(sentence_id_train, logits_list_train, y_pred_train, y_true_train, os.path.join(model_path, "predictions_train.tsv"))

            dict_of_results["TRAIN"] = {"F1": final_train_f1, "Accuracy": final_train_accuracy, "Loss": final_train_loss}

        return dict_of_results

    def train_and_evaluate_with_multiple_seeds(self, no_times, seeds_from_config=False, eval_on_train=True):
        accuracy_agg, model_pathes = self.train_with_multiple_seeds(no_times, seeds_from_config=seeds_from_config)

        dict_of_seed_results = {}
        aggregation_dict = {"TRAIN": {"Accuracy": [], "Loss": []}, "DEV": {"Accuracy": [], "Loss": []}, "TEST": {"Accuracy": [], "Loss": []}}
        for index, path in enumerate(model_pathes):
            dict_of_results = self.evaluate_from_path(path, evaluate_on_train=eval_on_train)
            dict_of_seed_results[f"seed_{index}"] = dict_of_results
            aggregation_dict = update_dict_of_agg(aggregation_dict, dict_of_results, eval_on_train=eval_on_train)

        # Add a Field of Aggregation
        split_list = ["DEV", "TEST"]
        if eval_on_train:
            split_list.append("TRAIN")
        for split in split_list:
            aggregation_dict[split]["Accuracy"] = np.mean(aggregation_dict[split]["Accuracy"])
            aggregation_dict[split]["Loss"] = np.mean(aggregation_dict[split]["Loss"])

        dict_of_seed_results["Agg"] = aggregation_dict

        save_json(os.path.join(self.configs["checkpointing_path"], "final_scores.json"), dict_of_seed_results)
        if self.configs["use_neptune"]:
            neptune.stop()
        return dict_of_seed_results


if __name__ == "__main__":

    import sys
    config_file_path = "config.yaml"  # sys.argv[1]

    trainer_class = Trainer(config_file_path=config_file_path)
    # trainer_class.train()
    # trainer_class.train_with_multiple_seeds(3)
    trainer_class.train_and_evaluate_with_multiple_seeds(1, seeds_from_config=True, eval_on_train=False)
