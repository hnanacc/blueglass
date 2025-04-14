class BestTracker:
    def __init__(self):
        self.best_fitness = None
        self.best_step = None

    def is_best(self, fitness: float, step: int) -> bool:
        is_best = self.best_fitness is None or fitness > self.best_fitness

        if is_best:
            self.best_fitness = fitness
            self.best_step = step

        return is_best

    def best(self):
        if self.best_fitness is not None:
            return self.best_fitness
        else:
            return 0