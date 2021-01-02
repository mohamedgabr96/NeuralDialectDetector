import torch
import math
from torch import nn
from transformers.models.bert.modeling_bert import BertSelfOutput, BertOutput, BertAttention, BertLayer


class AdapterModule(nn.Module):
    def __init__(self, hidden_size, bottleneck_dim):
        super(AdapterModule, self).__init__()
        self.down_project = nn.Linear(hidden_size, bottleneck_dim)
        self.up_project = nn.Linear(bottleneck_dim, hidden_size)
        self.non_linearity = nn.GELU()

    def forward(self, x):
        down_projection = self.down_project(x)
        non_linearity = self.non_linearity(down_projection)
        up_projection = self.up_project(non_linearity)

        return up_projection

## Inspired from fusion implementation in adapter-transformers repo https://github.com/Adapter-Hub/adapter-transformers
## From Class BERT Fusion implemented in: https://github.com/Adapter-Hub/adapter-transformers/blob/master/src/transformers/adapter_modeling.py

class FusionAttn(nn.Module):
    def __init__(self, config, bottleneck_dim, use_adapt_after_fusion):
        super(FusionAttn, self).__init__()

        self.config = config
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(config.hidden_size, 1)
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.T = 1.0
        self.reduction = self.T / 1000.0
        self.adapter_after_fusion = AdapterModule(config.hidden_size, bottleneck_dim*4)
        self.use_adapter_after_fusion = use_adapt_after_fusion
        
    def forward(self, hidden_states_before, adapters_output, attention_mask=None):
        key = adapters_output.permute(2, 1, 0, 3)
        value = adapters_output.permute(2, 1, 0, 3)
        query = hidden_states_before.permute(1, 0, 2)
        residual = hidden_states_before.permute(1, 0, 2)
    
        value += residual[:, :, None, :].repeat(1, 1, value.size(2), 1)

        query_layer = self.query(query)

        key_layer = self.key(key)
        value_layer = self.value(value)

        attention_scores = torch.squeeze(torch.matmul(query_layer.unsqueeze(2), key_layer.transpose(-2, -1)), dim=2)

        attention_scores = self.dropout(attention_scores)

        attention_probs = nn.Softmax(dim=-1)(attention_scores / self.T)
        self.T = max(self.T - self.reduction, 1.0)

        context_layer = torch.squeeze(torch.matmul(attention_probs.unsqueeze(2), value_layer), dim=2)

        context_layer = context_layer.permute(1, 0, 2)
        if self.use_adapter_after_fusion: 
            context_layer = self.adapter_after_fusion(context_layer)
        return context_layer


class AdapterFusionModule(nn.Module):
    def __init__(self, config, bottleneck_dim, current_adapter_to_train, no_total_adapters, stage_2_training, use_adapt_after_fusion):
        super(AdapterFusionModule, self).__init__()
        self.stage_2_training = stage_2_training
        self.adapter_to_train_index = current_adapter_to_train
        self.number_of_tasks = no_total_adapters
        self.list_of_adapter_modules = nn.ModuleList([AdapterModule(config.hidden_size, bottleneck_dim) for _ in range(self.number_of_tasks)]) 
        self.adapter_fusion_attention_layer = FusionAttn(config, bottleneck_dim, use_adapt_after_fusion)
     
        if self.stage_2_training:
            for p in self.list_of_adapter_modules.named_parameters():
                p[1].requires_grad = False
        else:
            for p in self.adapter_fusion_attention_layer.named_parameters():
                p[1].requires_grad = False
            for adapter_index in range(self.number_of_tasks):
                if adapter_index != self.adapter_to_train_index:
                    for p in self.list_of_adapter_modules[adapter_index].named_parameters():
                        p[1].requires_grad = False

    
    def forward(self, hidden_states, attention_mask=None):
        if not self.stage_2_training:
            return self.list_of_adapter_modules[self.adapter_to_train_index](hidden_states)
        else:
            adapters_output = []
            for adapter_layer in self.list_of_adapter_modules:
                output_adapter_layer = adapter_layer(hidden_states)
                adapters_output.append(output_adapter_layer)
            adapters_output = torch.stack(adapters_output)

            final_hidden_states = self.adapter_fusion_attention_layer(hidden_states, adapters_output)
            return final_hidden_states

