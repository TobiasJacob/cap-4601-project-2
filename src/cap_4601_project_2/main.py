# from pureModel.entityFinder import EntityFinder
# from pureModel.relFinder import RelFinder
from src.semevalModel.entityModel import SemevalModel as SemevalEntityModel
from src.semevalModel.relationModel import (
    AlbertForRelation as SemevalRelationModel,
)
from src.semevalModel.semevalDataset import relationTypes
from transformers.models.albert import AlbertTokenizer


def main() -> None:
    sentences = "A wheel is a part of a car. Hunger is caused by an empty stomach. The following information appeared in the notes to consolidated financial statements of some corporate annual reports."
    print(sentences)
    sentences = [s + "." for s in sentences.split(". ")]
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
    # entFin = EntityFinder("ace05", "cpu")
    # relFin = RelFinder("ace05", "cpu")
    # text = "Steve Jobs is the founder of Apple."
    # entities = entFin.getEntities(text)
    # entFin.printEntities(text, entities)
    # print(relFin.getRelations(text, entities))


main()
