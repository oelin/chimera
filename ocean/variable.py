from dataclasses import dataclass 


@dataclass
class Variable:

    data: np.array
    grad: np.array
    requires_grad: bool = False 
    back: callable = None
    
    def backward(self, grad = 1.) -> None:
        self.grad += grad 
        self.back and self.back()
        self.grad *= self.requires_grad
        

def variable(data, requires_grad=False, back=None):

    return Variable(
        np.array(data).astype(float),
        np.zeros_like(np.array(data)).astype(float),
        requires_grad,
        back
    )
