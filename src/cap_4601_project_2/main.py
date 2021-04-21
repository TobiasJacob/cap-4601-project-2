# from pureModel.entityFinder import EntityFinder
# from pureModel.relFinder import RelFinder
from src.semevalModel.entityModel import SemevalModel as SemevalEntityModel
from transformers.models.albert import AlbertTokenizer


def main() -> None:
    sentences = "A wheel is a part of a car. Hunger is caused by an empty stomach. The following information appeared in the notes to consolidated financial statements of some corporate annual reports."
    sentences = [s + "." for s in sentences.split(". ")]
    modelname = "models/Apr21_18-04-28_78c99689a048Entity/5"
    tokenizer = AlbertTokenizer.from_pretrained(modelname)

    model = SemevalEntityModel.from_pretrained(modelname)
    spans, tokens, cleartext = model.getEntities(tokenizer, sentences)
    print(cleartext)
    # entFin = EntityFinder("ace05", "cpu")
    # relFin = RelFinder("ace05", "cpu")
    # text = "Steve Jobs is the founder of Apple."
    # entities = entFin.getEntities(text)
    # entFin.printEntities(text, entities)
    # print(relFin.getRelations(text, entities))


main()
