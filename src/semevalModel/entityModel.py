import torch
from allennlp.nn.util import batched_index_select
from src.semevalModel.semevalDataset import getSpans
from torch import nn
from torch.nn import BCELoss
from torch.nn.utils.rnn import pad_sequence
from transformers import AlbertModel, AlbertPreTrainedModel


def f1_loss(
    y_true: torch.Tensor, y_pred: torch.Tensor, is_training=False
) -> torch.Tensor:
    tp = (y_true * y_pred).sum().to(torch.float32)
    # tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    # print(precision.item(), recall.item())
    return (f1, precision, recall)


class SemevalModel(AlbertPreTrainedModel):
    def __init__(
        self,
        config,
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

        self.layer_norm = nn.LayerNorm(
            config.hidden_size * 2 + width_embedding_dim
        )
        self.ner_classifier = nn.Sequential(
            nn.Linear(
                config.hidden_size * 2 + width_embedding_dim, head_hidden_dim
            ),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.LayerNorm(head_hidden_dim),
            nn.Linear(head_hidden_dim, head_hidden_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, 1),
        )

        self.init_weights()

    def getEntities(self, tokenizer, sentences, maxSpanLen=8):
        tokens = tokenizer(sentences)["input_ids"]
        input_ids = pad_sequence(
            [torch.tensor(t) for t in tokens], batch_first=True
        )
        spanBatch = [getSpans(t, maxSpanLen) for t in tokens]
        spanBatch = pad_sequence(spanBatch, batch_first=True)
        spanprop, _ = self(
            input_ids=input_ids,
            spans=spanBatch,
            spans_mask=spanBatch[:, :, 2] != 0,
            token_type_ids=(input_ids == 0).type(torch.int),
            attention_mask=(input_ids != 0).type(torch.int),
            spans_ner_label=None,
        )
        spanI1 = spanprop.argmax(1)
        spans1 = spanBatch[torch.arange(len(spanI1)), spanI1].clone()
        spanprop[torch.arange(len(spanI1)), spanI1] = -float("inf")
        spanI2 = spanprop.argmax(1)
        spans2 = spanBatch[torch.arange(len(spanI2)), spanI2]
        # spanBatch = spanBatch[spanprop > 0.5]
        spanBatch = torch.stack((spans1, spans2), dim=1)
        cleartext = [
            [
                tokenizer.convert_tokens_to_string(
                    tokenizer.convert_ids_to_tokens(
                        t[s[0].item() : s[1].item() + 1]
                    )
                )
                for s in spans
            ]
            for (t, spans) in zip(tokens, spanBatch)
        ]
        return spanBatch, tokens, cleartext

    def forward(
        self,
        input_ids,
        spans,
        spans_mask,
        token_type_ids=None,
        attention_mask=None,
        spans_ner_label=None,
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
        spans_start = spans[:, :, 0]
        spans_start_embedding = batched_index_select(
            sequence_output, spans_start
        )
        spans_end = spans[:, :, 1]
        spans_end_embedding = batched_index_select(sequence_output, spans_end)

        spans_width = spans[:, :, 2]
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

        logits = self.ner_classifier(spans_embedding)[:, :, 0]
        logits = torch.sigmoid(logits)

        if spans_ner_label is not None:
            gold = spans_ner_label.type(torch.float)
            pred = logits
            loss_fct = BCELoss(
                weight=(gold > 0.5) * 0.97 + 0.03, reduction="sum"
            )  # 0 is 98.56% of the time, 1 is 1.44% of the time
            loss = loss_fct(pred, gold)
            # print(gold[0], pred[0])

            return loss, f1_loss(gold, pred), logits, spans_embedding
        else:
            return logits, spans_embedding
