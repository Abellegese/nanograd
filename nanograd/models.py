import optimizers
from metrics import Loss
class Sequential:
    def __init__(self):
        self.stack = []
        self.optimizer_type = None
        self.loss_type = []
    def add(self, layers):
        self.stack.append(layers)
    def predict(network, input):
        output = input
        for layer in network:
            output = layer.forward(output)
        return output
    def train(self,  X, Y, loss, epochs=20, lr=0.01):
        loss, epoch, history = [], [], []
        cost, cost_prime = Loss.mse, Loss.mse_prime
        if loss == "binary_cross_entropy":
            cost = Loss.binary_cross_entropy
            cost_prime = Loss.binary_cross_entropy_prime
        for i in range(epochs):
            error = 0
            for x, y in zip(X, Y):
                #forward
                output = Sequential.predict(self.stack, x)
                #print(output)
                #calculate the error
                error += cost(y, output)
                loss.append(error)
                #print(error)
                #calculate the backprop
                grad = cost_prime(y, output)
                for layer in reversed(self.stack):
                    grad = layer.backward(grad, lr)
            epoch.append(i+1)
            error /= len(X)
            print(f"{i + 1}/{epochs}, error={error}")
        history.append(epoch)
        history.append(loss)
        return history

