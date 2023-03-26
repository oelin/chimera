from . optimizers import Optimizer
from dataclasses import dataclass


@dataclass
class GradientDescentOptimizer(Optimizer):
    learning_rate: float = 1e-3

    def step(self):
        for parameter in self.parameters:
            parameter.data -= parameter.grad * self.learning_rate
