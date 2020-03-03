

class Optimizer:
    pass

class RandomRestartOptimizer(Optimizer):
    def __init__(self, N=10):
        self.N=N