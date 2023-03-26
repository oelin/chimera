class LinearTransformation(Function):

    def forward(self, x: Variable, w: Variable, b: Variable) -> Variable:
        return variable(x.data @ w.data + b.data)
    
    def backward(self, x, w, b, grad):
        grad_x = np.dot(grad, w.data.T)
        grad_weights = np.dot(x.data.T, grad)
        grad_bias = np.sum(grad, axis=0)
        x.grad += grad_x
        w.grad += grad_weights
        b.grad += grad_bias


class Linear(Module):

    def __init__(self, in_features: int, out_features: int) -> None:
        self.weights = variable(data=np.random.randn(in_features, out_features), requires_grad=True)
        self.bias = variable(data=np.random.randn(out_features), requires_grad=True)
    
    @property
    def parameters(self):
        return self.weights, self.bias
    
    def forward(self, x: Variable) -> Variable:
        return LinearTransformation()(x, self.weights, self.bias)
