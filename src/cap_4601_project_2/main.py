from pureModel.entityFinder import EntityFinder
from pureModel.relFinder import RelFinder


def main() -> None:
    entFin = EntityFinder("ace05", "cpu")
    relFin = RelFinder("ace05", "cpu")
    text = "Steve Jobs is the founder of Apple."
    entities = entFin.getEntities(text)
    entFin.printEntities(text, entities)
    print(relFin.getRelations(text, entities))
