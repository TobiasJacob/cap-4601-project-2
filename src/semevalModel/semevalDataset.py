import re

import torch
from transformers.models.albert import AlbertTokenizer

datasetpath = "pretrained/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT"  # noqa: E501

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


class SemevalDataset(torch.utils.data.Dataset):
    def __init__(self):
        tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
        tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>"]}
        )
        lines = open(datasetpath).readlines()
        sentences = lines[0::4]
        sentences = [re.sub(r'^.*?"', "", l) for l in sentences]
        sentences = [re.sub(r'"\n', "", l) for l in sentences]
        self.sentences = [tokenizer(l)["input_ids"] for l in sentences]
        relations = lines[1::4]
        relations = [re.sub(r"\n", "", l) for l in relations]
        self.relations = [relationTypes.index(l) for l in relations]

    def __getitem__(self, index):
        return (self.sentences[index], self.relations[index])

    def __len__(self):
        return len(self.sentences)
