from model import GymModel
import model
import random
import threading

class GEntity:
    def __init__(self, entity_id, hyperparameters):
        self.entity_id = entity_id
        self.model = GymModel(entity_id, 100)
        self.hyperparameters = hyperparameters
        self.score = 0.

    def get_id(self):
        return self.entity_id

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

class EntityThread(threading.Thread):
    def __init__(self, entity, generation):
        threading.Thread.__init__(self)
        self.entity = entity
        self.generation = generation

    def run(self):
        print('[' + str(self.entity.get_id()) + '] Training entity...')
        self.entity.eval(self.generation)

def train(generations_count, entities_count, mutation_rate):
    entities = []
    for i in range(entities_count):
        entities.append(GEntity(i, [random.uniform(0., 1.) for _ in range(3)]))

    for g in range(generations_count):
        print('Training generation ' + str(i) + '...')

        threads = []
        for i in range(entities_count):
            t = EntityThread(entities[i], g)
            t.start()
            threads.append(t)

        for i in range(entities_count):
            threads[i].join()

        sorted(entities)
        entities[0:int(entities_count / 2)] = entities[int(entities_count / 2):entities_count]
        sorted(entities)

        for i in range(entities_count):
            if random.uniform(0., 1.) < mutation_rate:
                entities[i].mutate()
        # TODO Crossover
