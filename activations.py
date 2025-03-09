import numpy as np

class BaseActivationFunction:
    def __init__(self, name):
        self.name = name
    def forward(self):
        pass
    def backward(self):
        pass

class ReLU(BaseActivationFunction):
    def __init__(self, name="ReLU"):
        self.name = name
    def forward(self, A):
        return np.maximum(0, A)
    def backward(self, H):
        return np.diag(H>0)

class TanH(BaseActivationFunction):
    def __init__(self, name="TanH"):
        self.name = name
    def forward(self, A):
        return np.tanh(A)
    def backward(self, H):
        return np.diag(1-H**2)

class Softmax(BaseActivationFunction):
    def __init__(self, name="Softmax"):
        self.name = name
    def forward(self, A):
        exp_A = np.exp(A - np.max(A, axis=0, keepdims=True))
        return exp_A/np.sum(exp_A, axis=0, keepdims=True)
    def backward(self, H):
        return (np.diag(H) - np.outer(H, H))

class Sigmoid(BaseActivationFunction):
    def __init__(self, name="Sigmoid"):
        self.name = name
    def forward(self, A):
        return 1/(1+np.exp(-A))
    def backward(self, H):
        return np.diag(H*(1-H))
    
class Identity(BaseActivationFunction):
    def __init__(self, name="Identity"):
        self.name = name
    def forward(self, A):
        return A
    def backward(self, H):
        return np.eye(H.shape[0])

ACTIVATIONS = {"Softmax": Softmax(), "ReLU": ReLU(), "tanh": TanH(), "sigmoid": Sigmoid(), "identity": Identity()}

