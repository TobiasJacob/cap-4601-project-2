import torch
from src.pureModel.entityFinder import task_ner_labels
from src.pureModel.relModels import AlbertForRelation, BertForRelation
from transformers import AlbertTokenizer, BertTokenizer

task_rel_labels = {
    "ace04": ["PER-SOC", "OTHER-AFF", "ART", "GPE-AFF", "EMP-ORG", "PHYS"],
    "ace05": ["ART", "ORG-AFF", "GEN-AFF", "PHYS", "PER-SOC", "PART-WHOLE"],
    "scierc": [
        "PART-OF",
        "USED-FOR",
        "FEATURE-OF",
        "CONJUNCTION",
        "EVALUATE-FOR",
        "HYPONYM-OF",
        "COMPARE",
    ],
}


def add_marker_tokens(tokenizer, ner_labels):
    new_tokens = ["<SUBJ_START>", "<SUBJ_END>", "<OBJ_START>", "<OBJ_END>"]
    for label in ner_labels:
        new_tokens.append("<SUBJ_START=%s>" % label)
        new_tokens.append("<SUBJ_END=%s>" % label)
        new_tokens.append("<OBJ_START=%s>" % label)
        new_tokens.append("<OBJ_END=%s>" % label)
    for label in ner_labels:
        new_tokens.append("<SUBJ=%s>" % label)
        new_tokens.append("<OBJ=%s>" % label)
    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})


class RelFinder:
    def __init__(self, task, device):
        self.device = device
        self.task = task
        num_rel_labels = len(task_rel_labels[task]) + 1
        if task == "scierc":
            bert_model_name = "data/rel-scib-ctx0"
            self.rel_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
            self.rel_model = BertForRelation.from_pretrained(
                bert_model_name, num_rel_labels=num_rel_labels
            )
        else:
            bert_model_name = "data/rel-alb-ctx0"
            self.rel_tokenizer = AlbertTokenizer.from_pretrained(bert_model_name)
            self.rel_model = AlbertForRelation.from_pretrained(
                bert_model_name, num_rel_labels=num_rel_labels
            )
        add_marker_tokens(self.rel_tokenizer, task_ner_labels[task])
        self.rel_model.to(device)

    def generateRelationCandidates(self, text, entities):
        possibleRelations = []
        classes = task_ner_labels[self.task]
        tokens = self.rel_tokenizer.tokenize(text)
        for i in range(len(entities)):
            for j in range(len(entities)):
                if i != j:
                    subSpan, subClass = entities[i]
                    (subStart, subEnd) = (subSpan[0], subSpan[1])
                    ojbSpan, objClass = entities[j]
                    (objStart, objEnd) = (ojbSpan[0], ojbSpan[1])
                    newTokens = tokens.copy()
                    insertTokens = [
                        (
                            subStart - 1,
                            f"<SUBJ_START={classes[subClass - 1]}>",
                        ),
                        (subEnd, f"<SUBJ_END={classes[subClass - 1]}>"),
                        (objStart - 1, f"<OBJ_START={classes[objClass - 1]}>"),
                        (objEnd, f"<OBJ_END={classes[objClass - 1]}>"),
                    ]
                    insertTokens = list(reversed(sorted(insertTokens)))
                    for (index, insToken) in insertTokens:
                        newTokens.insert(index, insToken)
                    possibleRelations.append(
                        (
                            newTokens,
                            newTokens.index(f"<SUBJ_START={classes[subClass - 1]}>"),
                            newTokens.index(f"<OBJ_START={classes[objClass - 1]}>"),
                            f"<SUBJ_START={classes[subClass - 1]}>",
                            f"<SUBJ_END={classes[subClass - 1]}>",
                            f"<OBJ_START={classes[objClass - 1]}>",
                            f"<OBJ_END={classes[objClass - 1]}>",
                            classes[subClass - 1],
                            classes[objClass - 1],
                        )
                    )
        return possibleRelations

    def getRelations(self, text, entities):
        relationCandidates = self.generateRelationCandidates(text, entities)
        if len(relationCandidates) == 0:
            return []
        tokenized = []
        subjects = []
        objects = []
        sub_idx = []
        obj_idx = []
        for (
            tokens,
            subI,
            objI,
            subStartMarker,
            subEndMarker,
            objStartMarker,
            objEndMarker,
            subClass,
            objClass,
        ) in relationCandidates:
            with torch.no_grad():
                if self.task == "scierc":
                    tokenized.append(
                        [102] + self.rel_tokenizer.convert_tokens_to_ids(tokens) + [103]
                    )
                else:
                    tokenized.append(
                        [2] + self.rel_tokenizer.convert_tokens_to_ids(tokens) + [3]
                    )
                subjects.append(
                    self.rel_tokenizer.convert_tokens_to_string(
                        tokens[
                            tokens.index(subStartMarker)
                            + 1 : tokens.index(subEndMarker)
                        ]
                    )
                )
                objects.append(
                    self.rel_tokenizer.convert_tokens_to_string(
                        tokens[
                            tokens.index(objStartMarker)
                            + 1 : tokens.index(objEndMarker)
                        ]
                    )
                )
                sub_idx.append(subI + 1)
                obj_idx.append(objI + 1)

        tokenized = torch.tensor(tokenized).to(self.device)
        logits = self.rel_model(
            input_ids=tokenized,
            token_type_ids=torch.zeros_like(tokenized, device=self.device),
            attention_mask=torch.ones_like(tokenized, device=self.device),
            sub_idx=sub_idx,
            obj_idx=obj_idx,
        )

        relations = []
        relClasses = logits.argmax(1)
        for (relClass, subject, obj) in zip(relClasses, subjects, objects):
            if relClass > 0:
                relations.append((relClass.item(), subject, obj))
        return relations

    def printRelations(self, rels):
        for (relClass, subject, obj) in rels:
            print(
                task_rel_labels[self.task][relClass - 1]
                + "("
                + subject
                + ","
                + obj
                + ")"
            )
