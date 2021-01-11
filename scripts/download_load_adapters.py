import torch
from transformers import AutoTokenizer, AutoModel
import os
from tqdm import tqdm

PATH = "/Users/mohamedgabr/Documents/NeuralDialectDetector/pytorch_model"
PATH2 = "/Users/mohamedgabr/Documents/NeuralDialectDetector/pytorch_model_MABERT/pytorch_model.bin"
use_fusion = False
# model_name_on = "bashar-talafha/multi-dialect-bert-base-arabic"
model_name_on = "UBC-NLP/MARBERT"


tokenizer = AutoTokenizer.from_pretrained(model_name_on)
model = AutoModel.from_pretrained(model_name_on)

model.save_pretrained(PATH)
tokenizer.save_vocabulary(PATH)

model2 = AutoModel.from_pretrained(PATH)

# # load pytorch model of adapters
path2 = "/Users/mohamedgabr/Downloads/ar_houlsby_gelu/pytorch_adapter.bin"
model_adapters = torch.load(path2)

model2_state_dict = model2.state_dict()
model2_state_dict_2 = {}
for key, val in model2_state_dict.items():
    model2_state_dict_2["bert." + key] = val

num_adapters = 21

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

        # bert.encoder.layer.11.output.adapter_layer.adapter_fusion_attention_layer.adapter_after_fusion.down_project.bias
        # bert.encoder.layer.11.output.adapter_layer.list_of_adapter_modules.20.down_project.weight
        if use_fusion:
            for no_adapter in range(num_adapters):
                new_weight_key_attention = "bert.encoder.layer." + str(i) + ".attention.output.adapter_layer.list_of_adapter_modules." + str(no_adapter) + "." + str(new_updown) +  "_project.weight"
                new_bias_key_attention = "bert.encoder.layer." + str(i) + ".attention.output.adapter_layer.list_of_adapter_modules." + str(no_adapter) + "." + str(new_updown) +  "_project.bias"

                new_weight_key_output_layer = "bert.encoder.layer." + str(i) + ".output.adapter_layer.list_of_adapter_modules." + str(no_adapter) + "."  + str(new_updown) +  "_project.weight"
                new_bias_key_output_layer = "bert.encoder.layer." + str(i) + ".output.adapter_layer.list_of_adapter_modules." + str(no_adapter) + "."  + str(new_updown) +  "_project.bias"

                model2_state_dict_2[new_weight_key_attention] = model_adapters[weight_key_attention]
                model2_state_dict_2[new_bias_key_attention] = model_adapters[bias_key_attention]

                model2_state_dict_2[new_weight_key_output_layer] = model_adapters[weight_key_output_layer]
                model2_state_dict_2[new_bias_key_output_layer] = model_adapters[bias_key_output_layer]

            new_weight_key_attention = "bert.encoder.layer." + str(i) + ".attention.output.adapter_layer.adapter_fusion_attention_layer.adapter_after_fusion." + str(new_updown) +  "_project.weight"
            new_bias_key_attention = "bert.encoder.layer." + str(i) + ".attention.output.adapter_layer.adapter_fusion_attention_layer.adapter_after_fusion." + str(new_updown) +  "_project.bias"

            new_weight_key_output_layer = "bert.encoder.layer." + str(i) + ".output.adapter_layer.adapter_fusion_attention_layer.adapter_after_fusion."  + str(new_updown) +  "_project.weight"
            new_bias_key_output_layer = "bert.encoder.layer." + str(i) + ".output.adapter_layer.adapter_fusion_attention_layer.adapter_after_fusion."  + str(new_updown) +  "_project.bias"

            model2_state_dict_2[new_weight_key_attention] = model_adapters[weight_key_attention]
            model2_state_dict_2[new_bias_key_attention] = model_adapters[bias_key_attention]

            model2_state_dict_2[new_weight_key_output_layer] = model_adapters[weight_key_output_layer]
            model2_state_dict_2[new_bias_key_output_layer] = model_adapters[bias_key_output_layer]

        else:
            new_weight_key_attention = "bert.encoder.layer." + str(i) + ".attention.output.adapter_layer." + str(new_updown) +  "_project.weight"
            new_bias_key_attention = "bert.encoder.layer." + str(i) + ".attention.output.adapter_layer." + str(new_updown) +  "_project.bias"

            new_weight_key_output_layer = "bert.encoder.layer." + str(i) + ".output.adapter_layer."  + str(new_updown) +  "_project.weight"
            new_bias_key_output_layer = "bert.encoder.layer." + str(i) + ".output.adapter_layer."  + str(new_updown) +  "_project.bias"

            model2_state_dict_2[new_weight_key_attention] = model_adapters[weight_key_attention]
            model2_state_dict_2[new_bias_key_attention] = model_adapters[bias_key_attention]

            model2_state_dict_2[new_weight_key_output_layer] = model_adapters[weight_key_output_layer]
            model2_state_dict_2[new_bias_key_output_layer] = model_adapters[bias_key_output_layer]

 

# # add them model2

torch.save(model2_state_dict_2, PATH2)
print("Done")