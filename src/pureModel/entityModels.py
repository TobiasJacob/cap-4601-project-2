import logging

import torch
from allennlp.modules import FeedForward
from allennlp.nn.util import batched_index_select
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (
    AlbertModel,
    AlbertPreTrainedModel,
    BertModel,
    BertPreTrainedModel,
)

logger = logging.getLogger("root")


class BertForEntity(BertPreTrainedModel):
    def __init__(
        self,
        config,
        num_ner_labels,
        head_hidden_dim=150,
        width_embedding_dim=150,
        max_span_length=8,
    ):
        super().__init__(config)

        self.bert = BertModel(config)
        self.hidden_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.width_embedding = nn.Embedding(
            max_span_length + 1, width_embedding_dim
        )

        self.ner_classifier = nn.Sequential(
            FeedForward(
                input_dim=config.hidden_size * 2 + width_embedding_dim,
                num_layers=2,
                hidden_dims=head_hidden_dim,
                activations=nn.ReLU(),
                dropout=0.2,
            ),
            nn.Linear(head_hidden_dim, num_ner_labels),
        )

        self.init_weights()

    def _get_span_embeddings(
        self, input_ids, spans, token_type_ids=None, attention_mask=None
    ):
        sequence_output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )["last_hidden_state"]

        sequence_output = self.hidden_dropout(sequence_output)

        """
        spans: [batch_size, num_spans, 3]; 0: left_ned, 1: right_end, 2: width
        spans_mask: (batch_size, num_spans, )
        """
        spans_start = spans[:, :, 0].view(spans.size(0), -1)
        spans_start_embedding = batched_index_select(
            sequence_output, spans_start
        )
        spans_end = spans[:, :, 1].view(spans.size(0), -1)
        spans_end_embedding = batched_index_select(sequence_output, spans_end)

        spans_width = spans[:, :, 2].view(spans.size(0), -1)
        spans_width_embedding = self.width_embedding(spans_width)

        # Concatenate embeddings of left/right points and the width embedding
        spans_embedding = torch.cat(
            (
                spans_start_embedding,
                spans_end_embedding,
                spans_width_embedding,
            ),
            dim=-1,
        )
        """
        spans_embedding: (batch_size, num_spans, hidden_size*2+embedding_dim)
        """
        return spans_embedding

    def forward(
        self,
        input_ids,
        spans,
        spans_mask,
        spans_ner_label=None,
        token_type_ids=None,
        attention_mask=None,
    ):
        spans_embedding = self._get_span_embeddings(
            input_ids,
            spans,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        ffnn_hidden = []
        hidden = spans_embedding
        for layer in self.ner_classifier:
            hidden = layer(hidden)
            ffnn_hidden.append(hidden)
        logits = ffnn_hidden[-1]

        if spans_ner_label is not None:
            loss_fct = CrossEntropyLoss(reduction="sum")
            if attention_mask is not None:
                active_loss = spans_mask.view(-1) == 1
                active_logits = logits.view(-1, logits.shape[-1])
                active_labels = torch.where(
                    active_loss,
                    spans_ner_label.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(
                        spans_ner_label
                    ),
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(
                    logits.view(-1, logits.shape[-1]), spans_ner_label.view(-1)
                )
            return loss, logits, spans_embedding
        else:
            return logits, spans_embedding, spans_embedding


class AlbertForEntity(AlbertPreTrainedModel):
    def __init__(
        self,
        config,
        num_ner_labels,
        head_hidden_dim=150,
        width_embedding_dim=150,
        max_span_length=8,
    ):
        super().__init__(config)

        self.albert = AlbertModel(config)
        self.hidden_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.width_embedding = nn.Embedding(
            max_span_length + 1, width_embedding_dim
        )

        self.ner_classifier = nn.Sequential(
            FeedForward(
                input_dim=config.hidden_size * 2 + width_embedding_dim,
                num_layers=2,
                hidden_dims=head_hidden_dim,
                activations=nn.ReLU(),
                dropout=0.2,
            ),
            nn.Linear(head_hidden_dim, num_ner_labels),
        )

        self.init_weights()

    def _get_span_embeddings(
        self, input_ids, spans, token_type_ids=None, attention_mask=None
    ):
        sequence_output = self.albert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )["last_hidden_state"]

        sequence_output = self.hidden_dropout(sequence_output)

        """
        spans: [batch_size, num_spans, 3]; 0: left_ned, 1: right_end, 2: width
        spans_mask: (batch_size, num_spans, )
        """
        spans_start = spans[:, :, 0].view(spans.size(0), -1)
        spans_start_embedding = batched_index_select(
            sequence_output, spans_start
        )
        spans_end = spans[:, :, 1].view(spans.size(0), -1)
        spans_end_embedding = batched_index_select(sequence_output, spans_end)

        spans_width = spans[:, :, 2].view(spans.size(0), -1)
        spans_width_embedding = self.width_embedding(spans_width)

        spans_embedding = torch.cat(
            (
                spans_start_embedding,
                spans_end_embedding,
                spans_width_embedding,
            ),
            dim=-1,
        )
        """
        spans_embedding: (batch_size, num_spans, hidden_size*2+embedding_dim)
        """
        return spans_embedding

    def forward(
        self,
        input_ids,
        spans,
        spans_mask,
        spans_ner_label=None,
        token_type_ids=None,
        attention_mask=None,
    ):
        spans_embedding = self._get_span_embeddings(
            input_ids,
            spans,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        ffnn_hidden = []
        hidden = spans_embedding
        for layer in self.ner_classifier:
            hidden = layer(hidden)
            ffnn_hidden.append(hidden)
        logits = ffnn_hidden[-1]

        if spans_ner_label is not None:
            loss_fct = CrossEntropyLoss(reduction="sum")
            if attention_mask is not None:
                active_loss = spans_mask.view(-1) == 1
                active_logits = logits.view(-1, logits.shape[-1])
                active_labels = torch.where(
                    active_loss,
                    spans_ner_label.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(
                        spans_ner_label
                    ),
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(
                    logits.view(-1, logits.shape[-1]), spans_ner_label.view(-1)
                )
            return loss, logits, spans_embedding
        else:
            return logits, spans_embedding, spans_embedding
