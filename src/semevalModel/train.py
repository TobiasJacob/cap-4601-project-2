from itertools import chain

import torch
from src.semevalModel.entityModel import SemevalModel
from src.semevalModel.semevalDataset import EntityDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup


def collate_fn_padd(batch):
    input_ids = pad_sequence([b[0] for b in batch], batch_first=True)
    token_type_ids = (input_ids == 0).type(torch.int)
    attention_mask = (input_ids != 0).type(torch.int)
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
    epochs = 5
    device = "cuda"
    model = SemevalModel.from_pretrained("albert-base-v2")
    model.to(device)
    optim = AdamW(
        [
            {"params": model.albert.parameters(), "lr": 1e-3},
            {
                "params": chain(
                    model.width_embedding.parameters(),
                    model.ner_classifier.parameters(),
                ),
                "lr": 1e-2,
            },
        ]
    )
    # optim = torch.optim.Adagrad(model.parameters(recurse=True), lr=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optim, 100, epochs * 10000 / 32
    )

    dataset = EntityDataset("albert-base-v2", device=device)
    tokenTensor, spans, labels = next(iter(dataset))
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn_padd)

    for epoch in range(epochs):
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
                model.train()
                loss, f1, logits, spans_embedding = model(
                    input_ids,
                    spans,
                    spansMask,
                    spans_ner_label,
                    token_type_ids,
                    attention_mask,
                )
                loss.backward()
                optim.step()
                scheduler.step()
                optim.zero_grad()
                lossesTrain.append(loss.item())
                print(
                    "Train",
                    epoch,
                    i,
                    scheduler.get_lr(),
                    f1.item(),
                    lossesTrain[-1],
                )
            else:
                with torch.no_grad():
                    model.eval()
                    loss, f1, logits, spans_embedding = model(
                        input_ids,
                        spans,
                        spansMask,
                        spans_ner_label,
                        token_type_ids,
                        attention_mask,
                    )
                    lossesVal.append(loss.item())
                    print("Val", epoch, i, f1.item(), lossesVal[-1])
            i += 1


train()
