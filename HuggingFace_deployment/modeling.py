from transformers import BertPreTrainedModel, BertModel
from .modeling_utils import BertLayerPlainAdapters, ClassificationHead
from torch import nn
from .vertical_attention import SelfAttention as VerticalSelfAttention
from .config import NADIMARBERTConfig


class NADIMARBERTPreTrainedModel(BertPreTrainedModel):
    config_class = NADIMARBERTConfig
    base_model_prefix = "nadi_marbert"


class NADIMARBERTModelForSequenceClassification(NADIMARBERTPreTrainedModel):
    def __init__(self, config):
        super(NADIMARBERTModelForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config=config)
        self.bert.encoder.layer = nn.ModuleList([
            BertLayerPlainAdapters(config,
                config.bottleneck_dim)
                for _ in range(config.num_hidden_layers)])
        for param in self.bert.encoder.layer.named_parameters():
            if "adapter_layer" not in param[0]:
                param[1].requires_grad = False

        self.attend_vertical = VerticalSelfAttention(config)

        self.loss_function = nn.CrossEntropyLoss()
        self.classif_head = ClassificationHead(config.hidden_size, self.num_labels, config.classif_dropout_rate)

    def forward(self, input_ids, attention_mask, token_type_ids, class_label_ids):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        # sequence_output = outputs[0]  # Not needed for now
        pooled_output = outputs[1]  # [CLS]
        
        layer_cls = [layer[:,0,:] for layer in outputs[2]]
        pooled_output = self.attend_vertical(Xs=layer_cls, Q=pooled_output)

        logits = self.classif_head(pooled_output)

        total_loss = 0
        # 1. Intent Softmax
        if class_label_ids is not None:
            if self.num_labels == 1:
                loss_function = nn.MSELoss()
                temp_loss = loss_function(logits.view(-1), class_label_ids.view(-1))
            else:
                temp_loss = self.loss_function(logits.view(-1, self.num_labels), class_label_ids.view(-1))
            total_loss += temp_loss  

        outputs = ((logits, ),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits
