import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from semevalModel.entityModel import SemevalModel
from semevalModel.semevalDataset import EntityDataset, relationTypes


def collate_fn_padd(batch):
    input_ids = pad_sequence([b[0] for b in batch], batch_first=True)
    token_type_ids = (input_ids == 0).type(torch.int)
    attention_mask = torch.ones_like(input_ids)
    spans = pad_sequence(
        [b[1] for b in batch], batch_first=True, padding_value=0
    )
    spansMask = spans[:, :, 2] != 0
    spans_ner_label = pad_sequence(
        [b[2] for b in batch], batch_first=True, padding_value=0
    )

    return (
        input_ids,
        token_type_ids,
        attention_mask,
        spans,
        spansMask,
        spans_ner_label,
    )


def train():
    device = "cuda"
    model = SemevalModel.from_pretrained(
        "albert-base-v2", num_ner_labels=len(relationTypes)
    )
    model.to(device)
    optim = torch.optim.Adam(
        [
            {"params": model.albert.parameters()},
            {"params": model.width_embedding.parameters()},
            {"params": model.ner_classifier.parameters()},
        ]
    )

    dataset = EntityDataset("albert-base-v2", batch_size=32, device=device)
    tokenTensor, spans, labels = next(iter(dataset))
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn_padd)

    for epoch in range(5):
        i = 0
        lossesTrain = []
        lossesVal = []
        for (
            input_ids,
            token_type_ids,
            attention_mask,
            spans,
            spansMask,
            spans_ner_label,
        ) in dataloader:
            if i % 5 != 0:
                loss, logits, spans_embedding = model(
                    input_ids,
                    spans,
                    spansMask,
                    spans_ner_label,
                    token_type_ids,
                    attention_mask,
                )
                loss.backward()
                optim.step()
                lossesTrain.append(loss.item())
                print("Train", lossesTrain[-1])
            else:
                with torch.no_grad():
                    loss, logits, spans_embedding = model(
                        input_ids,
                        spans,
                        spansMask,
                        spans_ner_label,
                        token_type_ids,
                        attention_mask,
                    )
                    lossesVal.append(loss.item())
                    print("Val", lossesVal[-1])
            i += 1


train()
