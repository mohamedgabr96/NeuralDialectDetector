import yaml
import os
from tqdm import tqdm
import torch
import numpy as np
import json
import uuid


def read_yaml_file(file_path):
    with open(file_path, encoding="utf-8") as file_open:
        yaml_file = yaml.load(file_open, Loader=yaml.FullLoader)

    return yaml_file


def save_yaml_file(file_path, content):
    with open(file_path, encoding="utf-8", mode="w") as file_open:
        yaml_file = yaml.dump(content, file_open)


def save_model(model, tokenizer, path, used_config, step_no, current_dev_score=0):
    # final_path = os.path.join(path, f"checkpoint_w_dev_loss_{current_dev_score}_at_step_{step_no}_uuid_{uuid.uuid4().hex}")
    final_path = os.path.join(path, used_config["run_title"])
    if not os.path.exists(final_path):
        os.makedirs(final_path)
    model.save_pretrained(final_path)
    tokenizer.save_vocabulary(final_path)
    save_yaml_file(os.path.join(final_path, "config_model.yaml"), used_config)
    return final_path


def update_dict_of_agg(agg_dict, new_dict):
    agg_dict["TRAIN"]["Accuracy"].append(new_dict["TRAIN"]["Accuracy"])
    agg_dict["TRAIN"]["Loss"].append(new_dict["TRAIN"]["Loss"])

    agg_dict["DEV"]["Accuracy"].append(new_dict["DEV"]["Accuracy"])
    agg_dict["DEV"]["Loss"].append(new_dict["DEV"]["Loss"])

    agg_dict["TEST"]["Accuracy"].append(new_dict["TEST"]["Accuracy"])
    agg_dict["TEST"]["Loss"].append(new_dict["TEST"]["Loss"])
    return agg_dict


def save_json(path_to_file, content):
    with open(path_to_file, encoding="utf-8", mode="w") as ff:
        json.dump(content, ff)


def evaluate_predictions(model, evaluation_loader, model_class_name, device="cpu"):
    model.eval()
    no_batches = tqdm(evaluation_loader, desc="Batch Evaluation Loop")
    final_eval_loss, correct = 0, 0
    total_no_steps, num_samples = 0, 0
    for batch in no_batches:
        batch = [x.to(device) for x in batch]
        outputs = model(input_ids=batch[0], attention_mask=batch[1], token_type_ids=batch[2], class_label_ids=batch[3], input_ids_masked=batch[4])
        eval_loss, (logits,) = outputs[:2]
        final_eval_loss += eval_loss.mean().item()
                 
        total_no_steps += 1

        if model_class_name == "ArabicDialectBERT":
            label_ids = logits.argmax(axis=1)
            correct += (label_ids == batch[3]).sum()
            num_samples += label_ids.size(0)
    
    if model_class_name == "ArabicDialectBERT":
        accuracy = correct / float(num_samples)
        accuracy = accuracy.item()
    else:
        accuracy = 0
        
    eval_loss = final_eval_loss / total_no_steps

    return accuracy, eval_loss
