import torch
import math
from torch import nn
from transformers.models.bert.modeling_bert import BertSelfOutput, BertOutput, BertAttention, BertLayer
from AdaptersComponents.AdapterModules import AdapterFusionModule

# Taking each huggingface layer and overriding what should be overriden

class BertSelfOutput_w_adapters(BertSelfOutput):
    def __init__(self, config, bottleneck_dim, current_adapter_to_train, no_total_adapters, stage_2_training):
        super(BertSelfOutput_w_adapters, self).__init__(config)
        self.adapter_layer = AdapterFusionModule(config, bottleneck_dim, current_adapter_to_train, no_total_adapters, stage_2_training)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + self.adapter_layer(input_tensor) # Residual/Skip-Connection
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertOutput_w_adapters(BertOutput):
    def __init__(self, config, bottleneck_dim, current_adapter_to_train, no_total_adapters, stage_2_training):
        super(BertOutput_w_adapters, self).__init__(config)
        self.adapter_layer = AdapterFusionModule(config, bottleneck_dim, current_adapter_to_train, no_total_adapters, stage_2_training)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + self.adapter_layer(input_tensor)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class AdapterAttention(BertAttention):
    def __init__(self, config, bottleneck_dim, current_adapter_to_train, no_total_adapters, stage_2_training):
        super(AdapterAttention, self).__init__(config)
        self.output = BertSelfOutput_w_adapters(config, bottleneck_dim, current_adapter_to_train, no_total_adapters, stage_2_training) # Override with new SelfOut


class BertLayer_w_Adapters(BertLayer):
    def __init__(self, config, bottleneck_dim, current_adapter_to_train, no_total_adapters, stage_2_training):
        super(BertLayer_w_Adapters, self).__init__(config)
        self.attention = AdapterAttention(config, bottleneck_dim, current_adapter_to_train, no_total_adapters, stage_2_training)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = AdapterAttention(config, bottleneck_dim, current_adapter_to_train, no_total_adapters, stage_2_training)
        self.output = BertOutput_w_adapters(config, bottleneck_dim, current_adapter_to_train, no_total_adapters, stage_2_training)
