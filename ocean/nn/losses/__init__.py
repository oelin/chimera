from .. functions import Function


class MSELoss(Function):
    
    def forward(self, predicted: Variable, target: Variable) -> Variable:
        self.diff = predicted.data - target.data
        return variable(np.mean(np.square(self.diff)))
    
    def backward(self, predicted: Variable, target: Variable, gradient: np.array) -> None:
        mse_grad = 2 * self.diff / self.diff.size
        predicted.grad += mse_grad * gradient
        target.grad -= mse_grad * gradient

        
class BinaryCrossEntropyLoss(Function):
    
    def forward(self, y_pred: Variable, y_true: Variable) -> Variable:
        eps = 1e-15
        y_pred.data = np.clip(y_pred.data, eps, 1 - eps)
        loss = - (y_true.data * np.log(y_pred.data) + (1 - y_true.data) * np.log(1 - y_pred.data))
        return variable(np.mean(loss))

    def backward(self, y_pred: Variable, y_true: Variable, gradient: np.array) -> None:
        eps = 1e-15
        y_pred.data = np.clip(y_pred.data, eps, 1 - eps)
        n = y_pred.data.shape[0]
        dx = (y_pred.data - y_true.data) / (y_pred.data * (1 - y_pred.data) * n)
        y_pred.grad += dx * gradient
        y_true.grad -= dx * gradient
