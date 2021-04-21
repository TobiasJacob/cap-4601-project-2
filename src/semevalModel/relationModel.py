import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AlbertModel, AlbertPreTrainedModel


class AlbertForRelation(AlbertPreTrainedModel):
    def __init__(self, config, num_rel_labels):
        super(AlbertForRelation, self).__init__(config)
        self.num_labels = num_rel_labels
        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = torch.nn.LayerNorm(config.hidden_size * 2)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        sub_idx=None,
        obj_idx=None,
    ):
        outputs = self.albert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            output_attentions=False,
        )
        sequence_output = outputs[0]
        sub_output = torch.cat(
            [a[i].unsqueeze(0) for a, i in zip(sequence_output, sub_idx)]
        )
        obj_output = torch.cat(
            [a[i].unsqueeze(0) for a, i in zip(sequence_output, obj_idx)]
        )
        rep = torch.cat((sub_output, obj_output), dim=1)
        rep = self.layer_norm(rep)
        rep = self.dropout(rep)
        logits = self.classifier(rep)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            predictions = logits.argmax(1)
            acc = (predictions == labels).sum() / labels.shape[0]
            return loss, acc
        else:
            return logits
