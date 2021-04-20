from itertools import chain

import torch
from src.semevalModel.entityModel import SemevalModel
from src.semevalModel.semevalDataset import EntityDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
    writer = SummaryWriter()
    epochs = 5
    device = "cuda"
    model = SemevalModel.from_pretrained("albert-base-v2")
    model.to(device)
    optim = AdamW(
        [
            {"params": model.albert.parameters(), "lr": 1e-4},
            {
                "params": chain(
                    model.width_embedding.parameters(),
                    model.ner_classifier.parameters(),
                ),
                "lr": 1e-3,
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

    iTot = 0
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
                loss, (f1, precision, recall), logits, spans_embedding = model(
                    input_ids,
                    spans,
                    spansMask,
                    token_type_ids,
                    attention_mask,
                    spans_ner_label,
                )
                loss.backward()
                optim.step()
                scheduler.step()
                optim.zero_grad()
                lossesTrain.append(loss.item())
                writer.add_scalar("loss/Train", lossesTrain[-1], iTot)
                writer.add_scalar("f1/Train", f1.item(), iTot)
                writer.add_scalar("precision/Train", precision.item(), iTot)
                writer.add_scalar("recall/Train", recall.item(), iTot)
            else:
                with torch.no_grad():
                    model.eval()
                    (
                        loss,
                        (f1, precision, recall),
                        logits,
                        spans_embedding,
                    ) = model(
                        input_ids,
                        spans,
                        spansMask,
                        token_type_ids,
                        attention_mask,
                        spans_ner_label,
                    )
                    lossesVal.append(loss.item())
                    writer.add_scalar("f1/Eval", f1.item(), iTot)
                    writer.add_scalar("precision/Eval", precision.item(), iTot)
                    writer.add_scalar("recall/Eval", recall.item(), iTot)
                    writer.add_scalar("loss/Eval", lossesVal[-1], iTot)
            if iTot % 20 == 0:
                for (i2, lr) in enumerate(scheduler.get_lr()):
                    writer.add_scalar("lr/" + str(i2), lr, iTot)
            print(epoch, i)
            if iTot == 0:
                writer.add_graph(
                    model,
                    (
                        input_ids,
                        spans,
                        spansMask,
                        token_type_ids,
                        attention_mask,
                    ),
                )
                writer.flush()
            i += 1
            iTot += 1


train()
