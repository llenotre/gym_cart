from model import GymModel
import model
import random

class GEntity:
    def __init__(self, entity_id, hyperparameters):
        self.model = GymModel(entity_id, 300)
        self.hyperparameters = hyperparameters
        self.score = 0.

    def eval(self, generation_id):
        self.score = self.model.train(generation_id, self.hyperparameters, rendering=False)

    def get_score(self):
        return self.score

    def mutate(self):
        self.hyperparameters[int(random.uniform(0., 1.) * len(self.hyperparameters))] = random.uniform(0., 1.)

    def crossover(self, other):
        new_hyperparameters = self.hyperparameters
        for i in len(hyperparameters):
            if i % 2 == 0:
                new_hyperparameters[i] = other.hyperparameters[i]
        self.hyperparameters = new_hyperparameters

    def __eq__(self, other):
        return other.get_score() == self.get_score()

    def __ne__(self, other):
        return other.get_score() != self.get_score()

    def __lt__(self, other):
        return other.get_score() < self.get_score()

    def __le__(self, other):
        return other.get_score() <= self.get_score()

    def __gt__(self, other):
        return other.get_score() > self.get_score()

    def __ge__(self, other):
        return other.get_score() >= self.get_score()

def train(generations_count, entities_count, mutation_rate):
    entities = []
    for i in range(entities_count):
        entities.append(GEntity(i, [random.uniform(0., 1.) for _ in range(3)]))

    for g in range(generations_count):
        for i in range(entities_count):
            entities[i].eval(g) # TODO Multithread?

        sorted(entities)
        entities[0:int(entities_count / 2)] = entities[int(entities_count / 2):entities_count]
        sorted(entities)

        for i in range(entities_count):
            if random.uniform(0., 1.) < mutation_rate:
                entities[i].mutate()
        # TODO Crossover
