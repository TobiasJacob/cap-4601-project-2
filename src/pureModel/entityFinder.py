import torch
from src.pureModel.entityModels import AlbertForEntity, BertForEntity
from transformers import AlbertTokenizer, BertTokenizer

task_ner_labels = {
    "ace04": ["FAC", "WEA", "LOC", "VEH", "GPE", "ORG", "PER"],
    "ace05": ["FAC", "WEA", "LOC", "VEH", "GPE", "ORG", "PER"],
    "scierc": [
        "Method",
        "OtherScientificTerm",
        "Task",
        "Generic",
        "Material",
        "Metric",
    ],
}


class EntityFinder:
    def __init__(self, task, device):
        self.device = device
        self.task = task
        num_ner_labels = len(task_ner_labels[task]) + 1
        self.max_span_length = 8
        if task == "scierc":
            bert_model_name = "/workspaces/cap-4601-project-2/data/ent-scib-ctx0"
            self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
            self.bert_model = BertForEntity.from_pretrained(
                bert_model_name,
                num_ner_labels=num_ner_labels,
                max_span_length=self.max_span_length,
            )
        else:  # ace05
            bert_model_name = "/workspaces/cap-4601-project-2/data/ent-alb-ctx0"
            self.tokenizer = AlbertTokenizer.from_pretrained(bert_model_name)
            self.bert_model = AlbertForEntity.from_pretrained(
                bert_model_name,
                num_ner_labels=num_ner_labels,
                max_span_length=self.max_span_length,
            )
        self.bert_model.to(device)

    def getEntities(self, text):
        tokens = self.tokenizer(text)["input_ids"]
        tokens = torch.tensor([tokens], device=self.device)
        sentL = tokens.shape[1] - 2
        spans = []
        sent_start = 1
        for i in range(sentL):
            for j in range(i, min(sentL, i + self.max_span_length)):
                spans.append((i + sent_start, j + sent_start, j - i + 1))
        with torch.no_grad():
            ner_logits, spans_embedding, last_hidden = self.bert_model(
                input_ids=tokens,
                spans=torch.tensor([spans], device=self.device),
                spans_mask=None,
                spans_ner_label=None,
                token_type_ids=torch.zeros_like(tokens, device=self.device),
                attention_mask=torch.ones_like(tokens, device=self.device),
            )
            _, predicted_label = ner_logits.max(2)
            predicted_label = predicted_label.cpu().numpy()
            last_hidden = last_hidden.cpu().numpy()
        entities = [(spans[i], v) for i, v in enumerate(predicted_label[0]) if v != 0]
        return (entities, tokens)

    def printEntities(self, text, entities):
        tokens = self.tokenizer.tokenize(text)
        for (span, spanClass) in entities:
            print(
                tokens[span[0] - 1 : span[1]],
                task_ner_labels[self.task][spanClass - 1],
            )
