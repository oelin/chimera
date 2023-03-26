from .. function import Function


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

        
class Sigmoid(Function):
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, x: Variable) -> Variable:
        y = self.sigmoid(x.data)
        return variable(y)

    def backward(self, x: Variable, grad: np.array) -> None:
        y = self.sigmoid(x.data)
        dx = y * (1 - y) * grad
        x.grad += dx

sigmoid = Sigmoid()
linear = LinearTransformaton()
