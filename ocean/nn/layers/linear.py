from ... module import Module
from .. functions import linear


class Linear(Module):

    def __init__(self, in_features: int, out_features: int) -> None:
        self.weights = variable(data=np.random.randn(in_features, out_features), requires_grad=True)
        self.bias = variable(data=np.random.randn(out_features), requires_grad=True)
    
    @property
    def parameters(self):
        return self.weights, self.bias
    
    def forward(self, x: Variable) -> Variable:
        return linear(x, self.weights, self.bias)
