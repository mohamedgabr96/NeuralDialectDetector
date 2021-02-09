import numpy as np
import torch
from flask import Flask, request, render_template
import sys
sys.path.append("../../")
import model as model_classes
from transformers import AutoTokenizer, AutoModel, AutoConfig
from general_utils import read_yaml_file 
from dataset_utils import parse_classes_list, convert_examples_to_features
app = Flask(__name__)


# Initializing needed classes
global classes_list, tokenizer, model, configs
config_file_path = "../../config.yaml"
data_classes_path = "../../dataset_dummy"
classes_list = parse_classes_list(data_classes_path)
classes_list.sort()
configs = read_yaml_file(config_file_path)
configs["num_labels"] = len(classes_list)
configs["device"] = "cpu"
configs["use_vert_att"] = "true"
model_name_path = configs["model_name_path"]
tokenizer = AutoTokenizer.from_pretrained(model_name_path)
model_config = AutoConfig.from_pretrained(model_name_path)
configs["mask_id"] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
model = getattr(model_classes, configs["model_class"]).from_pretrained(model_name_path,
                                                        config=model_config,
                                                        args=configs)

model.to(configs["device"])

# General Functions
def return_logits(in_example):
    input_example = ("PREDICT_0", in_example, "None")
    max_seq_len = configs["max_sequence_length"]
    feats = convert_examples_to_features([input_example], classes_list, max_seq_len, tokenizer)[0]
    feats = [torch.tensor(x).unsqueeze(0) for x in feats]
    model_out = model(input_ids=feats[0], attention_mask=feats[1], token_type_ids=feats[2], class_label_ids=None, input_ids_masked=feats[4])
    logits_softmaxed = torch.nn.functional.softmax(model_out[1][0], dim=-1).detach().cpu().numpy()
    return logits_softmaxed


# Functions for the WebApp
@app.route('/', methods=["GET"])
def my_form():
    return render_template('first_page.html')


@app.route('/', methods=['POST'])
def my_form_post():
    input_text = request.form['text']
    logits_softmaxed = return_logits(input_text)
    return render_template('second_page.html', sentence_in=input_text, predict_country=str(classes_list[np.argmax(logits_softmaxed)]))


# APIs
# One function for country level, and another for the provincial
@app.route("/predict_class", methods=['GET', 'POST'])
def predict_class():
    content = request.json
    logits_softmaxed = return_logits(content["input_example"])
    return str(logits_softmaxed) + " " + str(classes_list[np.argmax(logits_softmaxed)])


