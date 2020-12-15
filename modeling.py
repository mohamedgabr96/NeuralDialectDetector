from transformers import AutoTokenizer, AutoModel, AutoConfig, AdamW, get_linear_schedule_with_warmup
from dataset_utils import parse_and_generate_loaders
from general_utils import read_yaml_file, save_model, evaluate_predictions
from model import ArabicDialectBERT
import logging
from tqdm import tqdm, trange
import numpy as np

logger = logging.getLogger(__name__)


tokenizer = AutoTokenizer.from_pretrained("bashar-talafha/multi-dialect-bert-base-arabic")
model_config = AutoConfig.from_pretrained("bashar-talafha/multi-dialect-bert-base-arabic")

configs = read_yaml_file("config.yaml")

# Generate Loaders
train_loader, dev_loader, test_loader, no_labels = parse_and_generate_loaders(configs["path_to_data"], tokenizer, batch_size=configs["batch_size"])
configs["num_labels"] = no_labels

# Instantiate Model
model = ArabicDialectBERT.from_pretrained("bashar-talafha/multi-dialect-bert-base-arabic",
                                                      config=model_config,
                                                      args=configs)


total_steps = len(train_loader) // configs["num_epochs"]

# Initialize Optimizers
optimizer = AdamW(model.parameters(), lr=configs["initial_learning_rate"], eps=configs["adam_epsilon"])
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=configs["warmup_steps"], num_training_steps=total_steps)


model.zero_grad()


# Training Loop
no_epochs = trange(configs["num_epochs"], desc="Epoch Number")
global_step = 0
training_loss = 0.0
best_dev_loss = np.inf
curr_dev_loss = np.inf
for _ in no_epochs:
    no_batches = tqdm(train_loader, desc="Batches Loop")
    for batch in no_batches:
        
        model.train()
        
        batch = [x.to(configs["device"]) for x in batch]
        outputs = model(input_ids=batch[0], attention_mask=batch[1], token_type_ids=batch[2], class_label_ids=batch[3])
        loss = outputs[0]


        loss.backward()

        training_loss += loss.item()

        optimizer.step()
        scheduler.step()  
        model.zero_grad()
        global_step += 1

        if global_step % configs["improvement_check_freq"] == 0:
            dev_accuracy, curr_dev_loss = evaluate_predictions(model, dev_loader, device=configs["device"])

        if configs["checkpoint_on_improvement"] and curr_dev_loss < best_dev_loss:
            logger.info(f"Dev Loss Reduction from {best_dev_loss} to {curr_dev_loss}")
            save_model(model, configs["checkpointing_path"], configs, step_no=global_step, current_dev_score=best_dev_loss)
            

        if configs["checkpointing_on"] and global_step % configs["checkpointing_freq"] == 0:
            logger.info(f"Checkpointing...")
            save_model(model, configs["checkpointing_path"], configs, step_no=global_step, current_dev_score=best_dev_loss)


loss_final = training_loss / global_step

# Final Evaluation Loop

final_dev_accuracy, final_dev_loss = evaluate_predictions(model, dev_loader, device=configs["device"])
final_test_accuracy, final_test_loss = evaluate_predictions(model, test_loader, device=configs["device"])

# Final Model Saving
if configs["save_final_model"]:
    save_model(model, configs["checkpointing_path"], configs, step_no=global_step, current_dev_score=final_dev_accuracy)



logger.info("Finished Training and Evaluation")

print("Done")