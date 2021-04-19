from semevalModel.semevalDataset import EntityDataset


def train():
    dataset = EntityDataset()
    tokenTensor, spans, labels = next(iter(dataset))
    print(list(zip(spans, labels)))
