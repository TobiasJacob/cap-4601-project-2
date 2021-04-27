from typing import List

import torch
from src.semevalModel.semevalDataset import relationTypes
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from transformers import AlbertModel, AlbertPreTrainedModel, AlbertTokenizer


class AlbertForRelation(AlbertPreTrainedModel):
    def __init__(self, config, num_rel_labels):
        super(AlbertForRelation, self).__init__(config)
        self.num_labels = num_rel_labels
        self.albert = AlbertModel(self.config)
        # self.albert.resize_token_embeddings(vocab_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.layer_norm = torch.nn.LayerNorm(self.config.hidden_size * 2)
        self.classifier = nn.Linear(self.config.hidden_size * 2, self.num_labels)
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

    def getRelations(
        self, tokenizer: AlbertTokenizer, tokenBatch, spanBatch: torch.Tensor
    ):
        # spanBatch[batchSize, 2, 3] # two most probable entities
        sentences = [None] * spanBatch.shape[0]
        sub_idx = [0] * spanBatch.shape[0]
        obj_idx = [0] * spanBatch.shape[0]

        markers = tokenizer.convert_tokens_to_ids(["<e1>", "</e1>", "<e2>", "</e2>"])
        for (i, (tokens, spans)) in enumerate(zip(tokenBatch, spanBatch)):
            i1S = spans[0, 0]
            i1E = spans[0, 1]
            i2S = spans[1, 0]
            i2E = spans[1, 1]
            if i1S > i2S:
                temp = (i2S, i2E)
                (i2S, i2E) = (i1S, i1E)
                (i1S, i1E) = temp
            tokens: List = tokens
            tokens.insert(i2E, markers[3])
            tokens.insert(i2S, markers[2])
            tokens.insert(i1E, markers[1])
            tokens.insert(i1S, markers[0])
            sentences[i] = torch.tensor(tokens)
            sub_idx[i] = i1S
            obj_idx[i] = i2S
        input_ids = pad_sequence(sentences, batch_first=True)
        relations = self(
            input_ids=input_ids,
            token_type_ids=(input_ids == 0).type(torch.int),
            attention_mask=(input_ids != 0).type(torch.int),
            labels=None,
            sub_idx=sub_idx,
            obj_idx=obj_idx,
        )
        relations = relations.argmax(1)
        return (relations, [relationTypes[rel] for rel in relations])
