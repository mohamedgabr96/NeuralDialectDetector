import torch.nn as nn
from transformers import AutoModel, PreTrainedModel


class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_labels, dropout_rate=0.):
        super(ClassificationHead, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_labels)  # Simple Head for Now

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class ArabicDialectBERT(PreTrainedModel):
    def __init__(self, config, args):
        super(ArabicDialectBERT, self).__init__(config, args)
        self.args = args
        self.num_labels = args["num_labels"]
        self.transformer_model = AutoModel.from_config(config)  # TO-DO: Add adapters function that changes encoder here # Load pretrained bert

        self.classif_head = ClassificationHead(config.hidden_size, self.num_labels, args["classif_dropout_rate"])

    def forward(self, input_ids, attention_mask, token_type_ids, class_label_ids):
        outputs = self.transformer_model(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]  # Not needed for now
        pooled_output = outputs[1]  # [CLS]

        logits = self.classif_head(pooled_output)

        total_loss = 0
        # 1. Intent Softmax
        if class_label_ids is not None:
            if self.num_labels == 1:
                loss_function = nn.MSELoss()
                temp_loss = loss_function(logits.view(-1), class_label_ids.view(-1))
            else:
                loss_function = nn.CrossEntropyLoss()
                temp_loss = loss_function(logits.view(-1, self.num_labels), class_label_ids.view(-1))
            total_loss += temp_loss  

        outputs = ((logits, ),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits