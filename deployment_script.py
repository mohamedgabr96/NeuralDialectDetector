import numpy as np
import torch
import sys
sys.path.append("./")
from azureml.core.model import Model
import os
import model as model_classes
from transformers import AutoTokenizer, AutoModel, AutoConfig
from general_utils import read_yaml_file 
from dataset_utils import parse_classes_list, convert_examples_to_features
import json


def init():
    global classes_list, tokenizer, model, configs
    model_name_id_AML = "neural_model_registration_2" # After you register the model to AML
    model_name_path = Model.get_model_path(model_name_id_AML)
    print("PATH TO MODEL")
    print(model_name_path)
    config_file_path = os.path.join(model_name_path, "config_model.yaml")
    data_classes_path = "/var/azureml-app/NeuralDialectDetectorWebApp/dataset_dummy"
    classes_list = parse_classes_list(data_classes_path)
    classes_list.sort()
    configs = read_yaml_file(config_file_path)
    configs["num_labels"] = len(classes_list)
    configs["device"] = "cpu"
    configs["use_vert_att"] = True
    configs["model_name_path"] = model_name_path
    tokenizer = AutoTokenizer.from_pretrained(model_name_path)
    model_config = AutoConfig.from_pretrained(model_name_path)
    configs["mask_id"] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    model = getattr(model_classes, configs["model_class"]).from_pretrained(model_name_path,
                                                            config=model_config,
                                                            args=configs)


def run(data):
    data_loaded = json.loads(data)
    input_sentence = data_loaded["input"]
    input_example = ("PREDICT_0", input_sentence, "None")
    max_seq_len = configs["max_sequence_length"]
    feats = convert_examples_to_features([input_example], classes_list, max_seq_len, tokenizer)[0]
    feats = [torch.tensor(x).unsqueeze(0) for x in feats]
    model_out = model(input_ids=feats[0], attention_mask=feats[1], token_type_ids=feats[2], class_label_ids=None, input_ids_masked=feats[4])
    logits_softmaxed = torch.nn.functional.softmax(model_out[1][0], dim=-1).detach().cpu().numpy()
    return {"country_prediction": str(classes_list[np.argmax(logits_softmaxed)]), "province_prediction": "None"}
