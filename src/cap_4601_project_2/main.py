from pretrainedEntity.entityFinder import EntityFinder


def main() -> None:
    entFin = EntityFinder("ace05", "cpu")
    text = "Steve Jobs is the founder of Apple."
    entities = entFin.getEntities(text)
    entFin.printEntities(text, entities)
