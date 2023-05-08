import numpy as np

class GD:
    def __init__(self,) -> None:
        pass
    def detrmine(og, lr, inputs, weight, bias, act='linear'):
        input_gradient = np.dot(np.transpose(weight), og)
        weight_gradient = np.dot(og, np.transpose(inputs))
        bias_gradient = og
        weight -= lr*weight_gradient
        bias -= lr*bias_gradient
        return input_gradient
        
class SGD:
    def __init__(self) -> None:
        pass