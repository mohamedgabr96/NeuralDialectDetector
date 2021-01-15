import torch 
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel
from transformers import BertModel, BertPreTrainedModel, BertForMaskedLM
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertLayer
from AdaptersComponents.BertLayerAdapter import BertLayer_w_Adapters
from AdaptersComponents.BertLayerPlainAdapter import BertLayer_w_PlainAdapters
from general_utils import random_mask_tokens


class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_labels, dropout_rate=0.):
        super(ClassificationHead, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_labels)  # Simple Head for Now

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class ArabicDialectBERT(BertPreTrainedModel):
    def __init__(self, config, args):
        super(ArabicDialectBERT, self).__init__(config, args)
        self.args = args
        self.num_labels = args["num_labels"]
        self.bert = BertModel(config=config)  # TO-DO: Add adapters function that changes encoder here # Load pretrained bert
        self.masking_perc = args["masking_percentage"]
        self.mask_id = args["mask_id"]
        self.device_name = args["device"]
        if args["use_adapters"]:
            if args["adapter_type"] == "Fusion":
                self.bert.encoder.layer = nn.ModuleList([BertLayer_w_Adapters(config, args["bottleneck_dim"], args["current_adapter_to_train"], args["no_total_adapters"], args["stage_2_training"], args["use_adapt_after_fusion"]) for _ in range(config.num_hidden_layers)])
                # self.bert.encoder.layer = nn.ModuleList([BertLayer(config) for _ in range(11)] + [BertLayer_w_Adapters(config, args["bottleneck_dim"], args["current_adapter_to_train"], args["no_total_adapters"], args["stage_2_training"], args["use_adapt_after_fusion"]) for _ in range(1)])
            elif args["adapter_type"] == "plain_adapter":
                self.bert.encoder.layer = nn.ModuleList([BertLayer_w_PlainAdapters(config, args["bottleneck_dim"], args["current_adapter_to_train"], args["no_total_adapters"], args["stage_2_training"], args["use_adapt_after_fusion"]) for _ in range(config.num_hidden_layers)])
                # self.bert.encoder.layer = nn.ModuleList([BertLayer(config) for _ in range(10)] + [BertLayer_w_PlainAdapters(config, args["bottleneck_dim"], args["current_adapter_to_train"], args["no_total_adapters"], args["stage_2_training"], args["use_adapt_after_fusion"]) for _ in range(2)])
            for param in self.bert.encoder.layer.named_parameters():
                if "adapter_layer" not in param[0]: #or "list_of_adapter_modules" in param[0]:
                    param[1].requires_grad = False
                else:
                    print(param[0])
            # Freeze all except adapters and head

        # self.loss_function = nn.CrossEntropyLoss(weight=torch.tensor(self.args["cls_weights"]))
        self.loss_function = nn.CrossEntropyLoss()
        self.classif_head = ClassificationHead(config.hidden_size, self.num_labels, args["classif_dropout_rate"])

    def forward(self, input_ids, attention_mask, token_type_ids, class_label_ids, input_ids_masked):
        if self.train:
            input_ids = random_mask_tokens(input_ids, attention_mask, self.masking_perc, self.mask_id, self.device_name)
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
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
                temp_loss = self.loss_function(logits.view(-1, self.num_labels), class_label_ids.view(-1))
            total_loss += temp_loss  

        outputs = ((logits, ),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits

class ArabicDialectBERTMaskedLM(BertForMaskedLM):
    def __init__(self, config, args):
        super(ArabicDialectBERTMaskedLM, self).__init__(config)
        self.args = args
        self.bert = BertModel(config, add_pooling_layer=False)

        if args["use_adapters"]:
            if args["adapter_type"] == "Fusion":
                self.bert.encoder.layer = nn.ModuleList([BertLayer_w_Adapters(config, args["bottleneck_dim"], args["current_adapter_to_train"], args["no_total_adapters"], args["stage_2_training"], args["use_adapt_after_fusion"]) for _ in range(config.num_hidden_layers)])
                # self.bert.encoder.layer = nn.ModuleList([BertLayer(config) for _ in range(11)] + [BertLayer_w_Adapters(config, args["bottleneck_dim"], args["current_adapter_to_train"], args["no_total_adapters"], args["stage_2_training"], args["use_adapt_after_fusion"]) for _ in range(1)])
            elif args["adapter_type"] == "plain_adapter":
                self.bert.encoder.layer = nn.ModuleList([BertLayer_w_PlainAdapters(config, args["bottleneck_dim"], args["current_adapter_to_train"], args["no_total_adapters"], args["stage_2_training"], args["use_adapt_after_fusion"]) for _ in range(config.num_hidden_layers)])
            for param in self.bert.encoder.layer.named_parameters():
                if "adapter_layer" not in param[0]:
                    param[1].requires_grad = False
            # Freeze all except adapters and head

        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

    def init_cls_weights(self):
        self._init_weights(self.cls.predictions.transform.dense)
        self._init_weights(self.cls.predictions.transform.LayerNorm)
        self._init_weights(self.cls.predictions.decoder)

    def forward(self, input_ids, attention_mask, token_type_ids, class_label_ids, input_ids_masked):

        outputs = self.bert(
            input_ids_masked,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if input_ids is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), input_ids.view(-1))

        outputs = ((prediction_scores, ),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (masked_lm_loss,) + outputs

        return outputs
