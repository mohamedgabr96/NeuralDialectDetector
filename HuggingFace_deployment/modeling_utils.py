from transformers.models.bert.modeling_bert import BertSelfOutput, BertOutput, BertAttention, BertLayer
from torch import nn


class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_labels, dropout_rate=0., depth=1):
        super(ClassificationHead, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.depth = depth
        assert depth in (1, 2)
        if depth == 1:
            self.linear = nn.Linear(input_dim, num_labels)
        else:
            self.linear = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                self.dropout,
                nn.Linear(input_dim // 2, num_labels)
            )

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

class AdapterModule(nn.Module):
    def __init__(self, hidden_size, bottleneck_dim):
        super(AdapterModule, self).__init__()
        self.down_project = nn.Linear(hidden_size, bottleneck_dim)
        self.up_project = nn.Linear(bottleneck_dim, hidden_size)
        self.non_linearity = nn.GELU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        down_projection = self.down_project(x)
        non_linearity = self.non_linearity(down_projection)
        up_projection = self.up_project(non_linearity)

        return up_projection


class BertOutputAdapters(BertOutput):
    def __init__(self, config, bottleneck_dim):
        super(BertOutputAdapters, self).__init__(config)
        self.adapter_layer = AdapterModule(config.hidden_size, bottleneck_dim)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + self.adapter_layer(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertSelfOutput_w_adapters(BertSelfOutput):
    def __init__(self, config, bottleneck_dim):
        super(BertSelfOutput_w_adapters, self).__init__(config)
        self.adapter_layer = AdapterModule(config.hidden_size, bottleneck_dim)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + self.adapter_layer(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class AdapterAttention(BertAttention):
    def __init__(self, config, bottleneck_dim):
        super(AdapterAttention, self).__init__(config)
        self.output = BertSelfOutput_w_adapters(config, bottleneck_dim)


class BertLayerPlainAdapters(BertLayer):
    def __init__(self, config, bottleneck_dim):
        super(BertLayerPlainAdapters, self).__init__(config)
        self.attention = AdapterAttention(config, bottleneck_dim)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = AdapterAttention(config, bottleneck_dim)
        self.output = BertOutputAdapters(config, bottleneck_dim)
