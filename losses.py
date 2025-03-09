import numpy as np

epsilon = 1e-6

class BaseLossFunction:
    def __init__(self, name):
        self.name = name
    def forward(self):
        pass
    def backward(self):
        pass

class MSE(BaseLossFunction):
    def __init__(self, name="MSE"):
        self.name = name
    def forward(self, y_hat, y):
        return np.mean((y_hat-y)**2)
    def backward(self, y_hat, y):
        return 2*(y_hat-y)
    
class CrossEntropy(BaseLossFunction):
    def __init__(self, name="Cross_Entropy"):
        self.name = name
    def forward(self, y_hat, y):
        return -np.dot(y, np.log(y_hat))
    def backward(self,y_hat, y):
        dH = np.zeros_like(y_hat)
        class_id = np.argmax(y)
        dH[class_id] = -1/(y_hat[class_id]+epsilon)
        return dH
    
LOSSES = {"mean_squarred_error": MSE, "cross_entropy":CrossEntropy}