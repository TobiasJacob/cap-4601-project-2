import torch
from src.semevalModel.relationModel import AlbertForRelation
from src.semevalModel.semevalDataset import RelationDataset, relationTypes
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup


def collate_fn_padd(batch):
    input_ids = pad_sequence([b[0] for b in batch], batch_first=True)
    token_type_ids = (input_ids == 0).type(torch.int)
    attention_mask = (input_ids != 0).type(torch.int)
    rel_label = torch.stack([b[1] for b in batch])
    e1_index = torch.stack([b[2] for b in batch])
    e2_index = torch.stack([b[3] for b in batch])

    return (
        input_ids,
        token_type_ids,
        attention_mask,
        rel_label,
        e1_index,
        e2_index,
    )


def train():
    writer = SummaryWriter(comment="Relation")
    modelDir = writer.log_dir.replace("runs", "models")
    epochs = 20
    device = "cuda"
    dataset = RelationDataset("albert-base-v2", device="cpu")
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn_padd)
    model = AlbertForRelation.from_pretrained(
        "albert-base-v2",
        num_rel_labels=len(relationTypes),
    )
    model.resize_token_embeddings(len(dataset.dataset.tokenizer))
    model.to(device)
    optim = AdamW(
        [
            {"params": model.albert.parameters(), "lr": 1e-4},
            {
                "params": model.classifier.parameters(),
                "lr": 1e-3,
            },
        ]
    )
    scheduler = get_linear_schedule_with_warmup(optim, 100, epochs * 10000 / 32)

    iTot = 0
    for epoch in range(epochs):
        i = 0
        lossesTrain = []
        lossesVal = []
        for (
            input_ids,
            token_type_ids,
            attention_mask,
            rel_label,
            e1_index,
            e2_index,
        ) in dataloader:
            if i % 5 != 0:
                model.train()
                loss, acc = model(
                    input_ids.to(device),
                    token_type_ids.to(device),
                    attention_mask.to(device),
                    rel_label.to(device),
                    e1_index.to(device),
                    e2_index.to(device),
                )
                loss.backward()
                optim.step()
                scheduler.step()
                optim.zero_grad()
                lossesTrain.append(loss.item())
                writer.add_scalar("lossRel/Train", lossesTrain[-1], iTot)
                writer.add_scalar("accRel/Train", acc.item(), iTot)
            else:
                with torch.no_grad():
                    model.eval()
                    loss, acc = model(
                        input_ids.to(device),
                        token_type_ids.to(device),
                        attention_mask.to(device),
                        rel_label.to(device),
                        e1_index.to(device),
                        e2_index.to(device),
                    )
                    lossesVal.append(loss.item())
                    writer.add_scalar("accRel/Eval", acc.item(), iTot)
                    writer.add_scalar("lossRel/Eval", lossesVal[-1], iTot)
            if iTot % 20 == 0:
                for (i2, lr) in enumerate(scheduler.get_lr()):
                    writer.add_scalar("lr/" + str(i2), lr, iTot)
            print(epoch, i)
            i += 1
            iTot += 1
        model.save_pretrained(modelDir + "/" + str(epoch))
        dataset.dataset.tokenizer.save_pretrained(modelDir + "/" + str(epoch))


train()
