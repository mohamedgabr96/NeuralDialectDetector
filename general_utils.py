import yaml
import os
from tqdm import tqdm
import torch
import numpy as np


def read_yaml_file(file_path):
    with open(file_path, encoding="utf-8") as file_open:
        yaml_file = yaml.load(file_open, Loader=yaml.FullLoader)

    return yaml_file


def save_yaml_file(file_path, content):
    with open(file_path, encoding="utf-8", mode="w") as file_open:
        yaml_file = yaml.dump(content, file_open)


def save_model(model, path, used_config, step_no, current_dev_score=0):
    final_path = os.path.join(path, f"checkpoint_w_dev_loss_{current_dev_score}_at_step_{step_no}")
    if not os.path.exists(final_path):
        os.makedirs(final_path)
    model.save_pretrained(final_path)
    save_yaml_file(os.path.join(final_path, "config_model.yaml"), used_config)


def evaluate_predictions(model, evaluation_loader, device="cpu"):
    model.eval()
    no_batches = tqdm(evaluation_loader, desc="Batch Evaluation Loop")
    final_eval_loss = 0
    total_no_steps = 0
    final_preds = None
    for batch in no_batches:
        with torch.no_grad():
            batch = [x.to(device) for x in batch]
            outputs = model(input_ids=batch[0], attention_mask=batch[1], token_type_ids=batch[2], class_label_ids=batch[3])
            eval_loss, (logits,) = outputs[:2]
            final_eval_loss += eval_loss.mean().item()
        
        total_no_steps += 1

        if final_preds is None:
            final_preds = logits.detach().cpu().numpy()
            label_ids = batch[3].detach().cpu().numpy()
        else:
            final_preds = np.append(final_preds, logits.detach().cpu().numpy(), axis=0)
            label_ids = np.append(label_ids, batch[3].detach().cpu().numpy(), axis=0)

    
    final_preds = np.argmax(final_preds, axis=1)
    accuracy = (final_preds == label_ids).mean()
    eval_loss = final_eval_loss / total_no_steps

    return accuracy, eval_loss
