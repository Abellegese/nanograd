import numpy as np
from optimizers import GD
from numpy.lib.stride_tricks import as_strided
class Layer:
    def __init__(self, input, output):
      self.input = input
      self.output = output
    def forward(self, input):
       pass
    def backward(self, output_gradient, lr):
       pass

class Dense(Layer):
    def __init__(self, input_size, output_size, activation = 'linear'):
        self.weight = np.zeros((output_size, input_size))
        self.bias = np.zeros((output_size, 1))
        self.activation = activation
    def forward(self, input):
        self.input = input
        return np.dot(self.weight, self.input) + self.bias

    def backward(self, output_gradient, lr):
        return GD.detrmine(output_gradient, lr, self.input, self.weight, self.bias, self.activation)

class Activation(Layer):

    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, lr):
        return np.multiply(output_gradient, self.activation_prime(self.input))
    
class Dropout(Layer):
    def __init__(self, input, output):
        super().__init__(input, output)
class Flatten(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)

class Conv2D(Layer):


    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height -
                             kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += np.correlate(self.input[j], self.kernels[i, j], "valid")
        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = np.correlate(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += np.convolve(output_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient
    
    class MaxPool2D(Layer):
        def __init__(self, stride, padding, kernel_size):
            self.stride = stride
            self.padding = padding
            self.kernel_size = kernel_size
        def forward(self, input):
            input = np.pad(input, self.padding, mode='constant')
    
            # Window view of input
            output_shape = ((input.shape[0] - self.kernel_size+2*self.padding)//self.stride + 1,
                            (input.shape[1] - self.kernel_size+2*self.padding)//self.stride + 1)
            shape_w = (output_shape[0], output_shape[1],
                       self.kernel_size, self.kernel_size)
            strides_w = (
                self.stride*input.strides[0], self.stride*input.strides[1], input.strides[0], input.strides[1])
            input_w = as_strided(input, shape_w, strides_w)
            
            # Return the result of pooling
            return input_w.max(axis=(2, 3))
        def backward(dout, A):
            # Get the shape and strides of A
            M,N = A.shape
            s0,s1 = A.strides
            
            # Get the kernel size and stride
            K,L = dout.shape
            S,T = M//K,N//L
            
            # Create an output array of zeros with the same shape as A
            dA = np.zeros_like(A)
            
            # Loop over the output array and set the gradient to the selected value
            for i in range(K):
                for j in range(L):
                    # Get the window view of A
                    window = A[i*S:(i+1)*S,j*T:(j+1)*T]
                    # Get the index of the maximum value
                    k,l = np.unravel_index(window.argmax(), window.shape)
                    # Set the gradient to the selected value
                    dA[i*S+k,j*T+l] = dout[i,j]
                    
            # Return the gradient of A
            return dA