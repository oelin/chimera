from dataclasses import dataclass


@dataclass
class Optimizer:
    parameters: tuple

    def zero_grads(self):
        for parameter in self.parameters:
            parameter.grad *= 0.
    
    def step(self):
        raise NotImplementedError()
