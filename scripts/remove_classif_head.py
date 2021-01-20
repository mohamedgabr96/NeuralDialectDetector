import torch
from transformers import AutoTokenizer, AutoModel
import os
from tqdm import tqdm

# PATH = "/Users/mohamedgabr/Documents/NeuralDialectDetector/pytorch_model"
# PATH2 = "/Users/mohamedgabr/Documents/NeuralDialectDetector/pytorch_model_2/pytorch_model.bin"

start_model_path = "./checkpoints_marbert/MARBERT_100/pytorch_model.bin"
end_model_path = "./checkpoints_marbert/MARBERT_100_1/pytorch_model.bin"


# # load pytorch model of adapters
model_with_head = torch.load(start_model_path)

## Delete classification head
model_with_head.pop('classif_head.linear.weight', None)
model_with_head.pop('classif_head.linear.bias', None)

## Save in new path
torch.save(model_with_head, end_model_path)
