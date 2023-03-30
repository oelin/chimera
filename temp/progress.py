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
    
    def __repr__(self):
        return f'Variable({self.data.round(4)}{(self.back and ", grad_fn=%s" % self.back) or ""}{self.requires_grad and ", requires_grad=True" or ""})'


@dataclass
class Function:
    def __call__(self, *variables) -> Variable:

        result = self.forward(*variables)

        def back():
            #print('here 1', result.grad)
            self.backward(*variables, result.grad)

            for variable in variables:
                variable.backward(0.)
        
        result.back = back 
        return result

def variable(data, requires_grad=False, back=None):

    return Variable(
        np.array(data).astype(float),
        np.zeros_like(np.array(data)).astype(float),
        requires_grad,
        back
    )
  
class Module:

    @property
    def parameters(self):
        return tuple()
    
    def __call__(self, *args):
        return self.forward(*args)

    def forward(self):
        raise NotImplementedError()
        
class LinearFunction(Function):

    def forward(self, x: Variable, w: Variable, b: Variable) -> Variable:
        return variable((x.data @ w.data.T) + b.data)
    
    def backward(self, x: Variable, w: Variable, b: Variable, grad) -> None:
        w.grad += grad.T @ x.data
        x.grad += grad @ w.data
        b.grad += np.sum(grad, axis=0)

linear = LinearFunction()


class Linear(Module):

    def __init__(self, in_features: int, out_features: int) -> None:
        self.weights = variable(data=np.random.randn(out_features, in_features), requires_grad=True)
        self.bias = variable(data=np.random.randn(out_features), requires_grad=True)
    
    @property
    def parameters(self):
        return self.weights, self.bias
    
    def forward(self, x: Variable) -> Variable:
        return linear(x, self.weights, self.bias)

class ReLU(Function):
    
    def forward(self, x: Variable) -> Variable:
        return variable(x.data * (x.data > 0))
    
    def backward(self, x: Variable, grad) -> None:
        x.grad += grad * (x.data > 0)

class MSELoss(Function):

    def forward(self, y_pred: Variable, y_true: Variable) -> Variable:
        assert y_pred.data.shape == y_true.data.shape
        self.diff = y_pred.data - y_true.data
        loss = variable(np.mean(self.diff**2))
        return loss
    
    def backward(self, y_pred: Variable, y_true: Variable, grad) -> None:
        y_pred.grad += (grad * 2 * self.diff) / self.diff.size 

from torch.nn import MSELoss as MSELossT
