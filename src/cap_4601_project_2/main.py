import torch
from src.pureModel.entityFinder import EntityFinder
from src.pureModel.relFinder import RelFinder
from src.semevalModel.entityModel import SemevalModel as SemevalEntityModel
from src.semevalModel.relationModel import (
    AlbertForRelation as SemevalRelationModel,
)
from src.semevalModel.semevalDataset import relationTypes
from transformers.models.albert import AlbertTokenizer


def main() -> None:
    sentencesJoined = "The bread is made with milk. "
    print(sentencesJoined)
    with torch.no_grad():
        # Model 1
        entFin = EntityFinder("ace05", "cuda")
        relFin = RelFinder("ace05", "cuda")
        text = sentencesJoined
        (entities, _) = entFin.getEntities(text)
        entFin.printEntities(text, entities)
        rels = relFin.getRelations(text, entities)
        relFin.printRelations(rels)

        # Model 3
        sentences = [s + "." for s in sentencesJoined.split(". ")]
        modelname = "models/Apr21_18-04-28_78c99689a048Entity/5"
        tokenizer = AlbertTokenizer.from_pretrained(modelname)

        model = SemevalEntityModel.from_pretrained(modelname)
        spans, tokens, cleartext = model.getEntities(tokenizer, sentences)
        print(cleartext)

        relmodelname = "models/Apr21_18-39-46_78c99689a048Relation/4"
        tokenizer = AlbertTokenizer.from_pretrained(relmodelname)
        model = SemevalRelationModel.from_pretrained(
            relmodelname,
            num_rel_labels=len(relationTypes),
        )
        relations, cleartextRels = model.getRelations(tokenizer, tokens, spans)
        print(cleartextRels)
        print("Extracted relations")
        for (entities, rel) in zip(cleartext, cleartextRels):
            print(rel.replace("e1", entities[0]).replace("e2", entities[1]))


main()
