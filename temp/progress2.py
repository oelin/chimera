import numpy as np


class Variable:
    def __init__(self, data, terminal=False):
        
        self.data = np.array(data)                        # x.
        self.grad = np.zeros_like(self.data)              # dL/dx.
        self.terminal = terminal
        self.backward = int 


class Function:
    def __call__(self, *variables):
        result = self.forward(*variables)                 # Forward pass, f(x0, ..., xn).
        
        def backward(grad = 1):                           # Backward pass.
            self.backward(*variables, grad + result.grad) # Accumulate gradients, dL/dxi += dL/df * df/dxi.

            for variable in variables:                    # Recurse.
                variable.backward(0)
                
            result.grad *= result.terminal                # Reset gradients.
        result.backward = backward 
        
        return result

      
class Module:
    def __call__(self, *variables):
        return self.forward(*variables)
    
    @property
    def variables(self):
        for variable in self.__dict__.values():
            if isinstance(variable, Module):
                yield from variable.variables
            
            if isinstance(variable, Variable):
                if variable.terminal:
                    yield variable
