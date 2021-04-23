import re
from typing import List, Tuple

import torch
from transformers.models.albert import AlbertTokenizer

datasetpathTrain = "pretrained/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT"  # noqa: E501
datasetpathTest = "pretrained/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT"  # noqa: E501

relationTypes = [
    "Cause-Effect(e1,e2)",
    "Instrument-Agency(e1,e2)",
    "Product-Producer(e1,e2)",
    "Content-Container(e1,e2)",
    "Entity-Origin(e1,e2)",
    "Entity-Destination(e1,e2)",
    "Component-Whole(e1,e2)",
    "Member-Collection(e1,e2)",
    "Message-Topic(e1,e2)",
    "Cause-Effect(e2,e1)",
    "Instrument-Agency(e2,e1)",
    "Product-Producer(e2,e1)",
    "Content-Container(e2,e1)",
    "Entity-Origin(e2,e1)",
    "Entity-Destination(e2,e1)",
    "Component-Whole(e2,e1)",
    "Member-Collection(e2,e1)",
    "Message-Topic(e2,e1)",
    "Other",
]


def getSpans(tokens: List[str], maxSpanLen: int) -> List[Tuple[int, int, int]]:
    tokensL = len(tokens)
    numSpans = 0
    for spanL in range(1, min(tokensL + 1, 1 + maxSpanLen)):
        numSpans += tokensL - spanL + 1
    spans = torch.empty((numSpans, 3), dtype=int)
    spanI = 0
    for i in range(tokensL):
        for j in range(i, min(tokensL, i + maxSpanLen)):
            spans[spanI, 0] = i
            spans[spanI, 1] = j
            spans[spanI, 2] = j - i + 1
            spanI += 1
    return spans


class SemevalDataset(torch.utils.data.Dataset):
    def __init__(self, modelname):
        self.tokenizer = AlbertTokenizer.from_pretrained(modelname)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>"]}
        )
        lines1 = open(datasetpathTrain).readlines()
        lines2 = open(datasetpathTest).readlines()
        sentences = lines1[0::4] + lines2[0::4]
        sentences = [re.sub(r'^.*?"', "", l) for l in sentences]
        sentences = [re.sub(r'"\n', "", l) for l in sentences]
        self.sentences = [self.tokenizer(l)["input_ids"] for l in sentences]
        relations = lines1[1::4] + lines2[1::4]
        relations = [re.sub(r"\n", "", l) for l in relations]
        self.relations = [relationTypes.index(l) for l in relations]

    def __getitem__(self, index: int) -> Tuple[List[int], int]:
        return (self.sentences[index], self.relations[index])

    def __len__(self) -> int:
        return len(self.sentences)


class EntityDataset(torch.utils.data.IterableDataset):
    def __init__(self, modelname, device="cpu"):
        self.dataset = SemevalDataset(modelname)
        self.device = device

    def __iter__(self):
        maxSpanLen = 7

        markers = self.dataset.tokenizer.convert_tokens_to_ids(
            ["<e1>", "</e1>", "<e2>", "</e2>"]
        )
        for sent, _ in self.dataset:
            tokenTensor = [t for t in sent if t not in markers]
            e1 = sent[sent.index(markers[0]) + 1 : sent.index(markers[1])]

            e2 = sent[sent.index(markers[2]) + 1 : sent.index(markers[3])]

            if len(e1) > maxSpanLen or len(e2) > maxSpanLen:
                print("Warning, long entity in ", sent, len(e1), len(e2))

            spans = getSpans(tokenTensor, maxSpanLen)
            tokenSpans = [tokenTensor[span[0] : span[1] + 1] for span in spans]

            labels = [1 if ts == e1 or ts == e2 else 0 for ts in tokenSpans]
            # labels = [1 if ts[2] == 1 else 0 for ts in spans]

            yield (
                torch.tensor(tokenTensor, device=self.device),
                spans.to(self.device),
                torch.tensor(
                    labels,
                    device=self.device,
                ),
            )


class RelationDataset(torch.utils.data.IterableDataset):
    def __init__(self, modelname, device="cpu"):
        self.dataset = SemevalDataset(modelname)
        self.device = device

    def __iter__(self):
        markers = self.dataset.tokenizer.convert_tokens_to_ids(
            ["<e1>", "</e1>", "<e2>", "</e2>"]
        )
        for sent, rel in self.dataset:
            yield (
                torch.tensor(sent, device=self.device),
                torch.tensor(rel, device=self.device),
                torch.tensor(sent.index(markers[0]), device=self.device),
                torch.tensor(sent.index(markers[2]), device=self.device),
            )
