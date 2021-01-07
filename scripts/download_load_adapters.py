import torch
from transformers import AutoTokenizer, AutoModel
import os
from tqdm import tqdm

PATH = "/Users/mohamedgabr/Documents/NeuralDialectDetector/pytorch_model"
PATH2 = "/Users/mohamedgabr/Documents/NeuralDialectDetector/pytorch_model_2/pytorch_model.bin"


tokenizer = AutoTokenizer.from_pretrained("bashar-talafha/multi-dialect-bert-base-arabic")
model = AutoModel.from_pretrained("bashar-talafha/multi-dialect-bert-base-arabic")

model.save_pretrained(PATH)
tokenizer.save_vocabulary(PATH)

model2 = AutoModel.from_pretrained(PATH)

# # load pytorch model of adapters
path2 = "/Users/mohamedgabr/Downloads/ar_houlsby_gelu/pytorch_adapter.bin"
model_adapters = torch.load(path2)

model2_state_dict = model2.state_dict()

for i in tqdm(range(12)):
    for updown in ["up", "down.0"]:
        weight_key_attention = "encoder.layer." + str(i) + ".attention.output.attention_text_lang_adapters.ar.adapter_" + str(updown) + "." + "weight"   
        bias_key_attention = "encoder.layer." + str(i) + ".attention.output.attention_text_lang_adapters.ar.adapter_" + str(updown) + "." + "bias"    

        weight_key_output_layer = "encoder.layer." + str(i) + ".output.layer_text_lang_adapters.ar.adapter_" + str(updown) + "." + "weight"
        bias_key_output_layer = "encoder.layer." + str(i) + ".output.layer_text_lang_adapters.ar.adapter_" + str(updown) + "." + "bias"

        if updown == "down.0":
            new_updown = "down"
        else:
            new_updown = updown

        new_weight_key_attention = "bert.encoder.layer." + str(i) + ".attention.output.adapter_layer." + str(new_updown) +  "_project.weight"
        new_bias_key_attention = "bert.encoder.layer." + str(i) + ".attention.output.adapter_layer." + str(new_updown) +  "_project.bias"

        new_weight_key_output_layer = "bert.encoder.layer." + str(i) + ".output.adapter_layer." + str(new_updown) +  "_project.weight"
        new_bias_key_output_layer = "bert.encoder.layer." + str(i) + ".output.adapter_layer." + str(new_updown) +  "_project.bias"

        model2_state_dict[new_weight_key_attention] = model_adapters[weight_key_attention]
        model2_state_dict[new_bias_key_attention] = model_adapters[bias_key_attention]

        model2_state_dict[new_weight_key_output_layer] = model_adapters[weight_key_output_layer]
        model2_state_dict[new_bias_key_output_layer] = model_adapters[bias_key_output_layer]

 

# # add them model2

torch.save(model2_state_dict, PATH2)
print("Done")