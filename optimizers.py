import numpy as np

epsilon = 1e-6

class BaseOptimizer:
    def __init__(self,name, lr):
        self.name = name
        self.lr = lr
    def update(self):
        pass

class SGD(BaseOptimizer):
    def __init__(self, weights=None, biases=None, lr=1e-3, name="SGD"):
        super().__init__(name, lr)
        self.weights = weights
        self.biases = biases
    def update(self, dW, dB):
        for layer in range(len(self.weights)):
            self.weights[layer] -= self.lr*dW[layer]
            self.biases[layer] -= self.lr*dB[layer]

class Momentum(BaseOptimizer):
    def __init__(self, weights=None, biases=None, lr=1e-3, momentum=0.9, name="Momentum"):
        super().__init__(name, lr)
        self.weights = weights
        self.biases = biases
        self.momentum = momentum
        self.vW = [np.zeros_like(W) for W in weights]
        self.vB = [np.zeros_like(B) for B in biases]
    def update(self, dW, dB):
        for layer in range(len(self.weights)):
            self.vW[layer] = self.momentum*self.vW[layer] + self.lr*dW[layer]
            self.vB[layer] = self.momentum*self.vB[layer] + self.lr*dB[layer]
            self.weights[layer] -= self.vW[layer]
            self.biases[layer] -= self.vB[layer]

class NestrovMomentum(BaseOptimizer):
    def __init__(self, weights=None, biases=None, lr=1e-3, momentum=0.9, name="NestrovMomentum"):
        super().__init__(name, lr)
        self.weights = weights
        self.biases = biases
        self.momentum = momentum
        self.vW = [np.zeros_like(W) for W in weights]
        self.vB = [np.zeros_like(B) for B in biases]
    def update(self, dW, dB):
        for layer in range(len(self.weights)):
            self.vW[layer] = self.momentum*self.vW[layer] + self.lr*dW[layer]
            self.vB[layer] = self.momentum*self.vB[layer] + self.lr*dB[layer]
            self.weights[layer] -= self.momentum*self.vW[layer] + self.lr*dW[layer]
            self.biases[layer] -= self.momentum*self.vB[layer] + self.lr*dB[layer]

class RMSProp(BaseOptimizer):
    def __init__(self, weights=None, biases=None, lr=1e-3, beta=0.9, epsilon=1e-6, name="RMSProp"):
        super().__init__(name, lr)
        self.weights = weights
        self.biases = biases
        self.beta = beta
        self.epsilon = epsilon
        self.sW = [np.zeros_like(W) for W in weights]
        self.sB = [np.zeros_like(B) for B in biases]
    def update(self, dW, dB):
        for layer in range(len(self.weights)):
            self.sW[layer] = self.beta*self.sW[layer] + (1-self.beta)*np.square(dW[layer])
            self.sB[layer] = self.beta*self.sB[layer] + (1-self.beta)*np.square(dB[layer])
            self.weights[layer] -= self.lr*dW[layer]/(np.sqrt(self.sW[layer])+self.epsilon)
            self.biases[layer] -= self.lr*dB[layer]/(np.sqrt(self.sB[layer])+self.epsilon)

class Adam(BaseOptimizer):
    def __init__(self, weights=None, biases=None, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-6, name="Adam"):
        super().__init__(name, lr)
        self.weights = weights
        self.biases = biases
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.mW = [np.zeros_like(W) for W in weights]
        self.vW = [np.zeros_like(W) for W in weights]
        self.mB = [np.zeros_like(B) for B in biases]
        self.vB = [np.zeros_like(B) for B in biases]
        self.steps = 0
    def update(self, dW, dB):
        self.steps += 1
        for layer in range(len(self.weights)):
            self.mW[layer] = self.beta1*self.mW[layer] + (1-self.beta1)*dW[layer]
            self.vW[layer] = self.beta2*self.vW[layer] + (1-self.beta2)*np.square(dW[layer])
            self.mB[layer] = self.beta1*self.mB[layer] + (1-self.beta1)*dB[layer]
            self.vB[layer] = self.beta2*self.vB[layer] + (1-self.beta2)*np.square(dB[layer])
            mW_hat = self.mW[layer]/(1-np.power(self.beta1,self.steps))
            vW_hat = self.vW[layer]/(1-np.power(self.beta2,self.steps))
            mB_hat = self.mB[layer]/(1-np.power(self.beta1,self.steps))
            vB_hat = self.vB[layer]/(1-np.power(self.beta2,self.steps))

            self.weights[layer] -= self.lr*mW_hat/(np.sqrt(vW_hat)+self.epsilon)
            self.biases[layer] -= self.lr*mB_hat/(np.sqrt(vB_hat)+self.epsilon)

class Nadam(BaseOptimizer):
    def __init__(self, weights=None, biases=None, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-6, name="Nadam"):
        super().__init__(name, lr)
        self.weights = weights
        self.biases = biases
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.mW = [np.zeros_like(W) for W in weights]
        self.vW = [np.zeros_like(W) for W in weights]
        self.mB = [np.zeros_like(B) for B in biases]
        self.vB = [np.zeros_like(B) for B in biases]
        self.steps = 0
    def update(self, dW, dB):
        self.steps += 1
        for layer in range(len(self.weights)):
            self.mW[layer] = self.beta1*self.mW[layer] + (1-self.beta1)*dW[layer]
            self.vW[layer] = self.beta2*self.vW[layer] + (1-self.beta2)*np.square(dW[layer])
            self.mB[layer] = self.beta1*self.mB[layer] + (1-self.beta1)*dB[layer]
            self.vB[layer] = self.beta2*self.vB[layer] + (1-self.beta2)*np.square(dB[layer])
            mW_hat = self.mW[layer]/(1-np.power(self.beta1,self.steps))
            vW_hat = self.vW[layer]/(1-np.power(self.beta2,self.steps))
            mB_hat = self.mB[layer]/(1-np.power(self.beta1,self.steps))
            vB_hat = self.vB[layer]/(1-np.power(self.beta2,self.steps))

            self.weights[layer] -= self.lr*(self.beta1*mW_hat/(np.sqrt(vW_hat)+self.epsilon) + (1-self.beta1)*dW[layer])
            self.biases[layer] -= self.lr*(self.beta1*mB_hat/(np.sqrt(vB_hat)+self.epsilon) + (1-self.beta1)*dB[layer])

def create_optimizer(args, weights, biases):
    if args.optimizer == "sgd":
        return SGD(weights,biases,lr=args.learning_rate)
    elif args.optimizer == "momentum":
        return Momentum(weights,biases,lr=args.learning_rate,momentum=args.momentum)
    elif args.optimizer == "nag":
        return NestrovMomentum(weights,biases,lr=args.learning_rate,momentum=args.momentum)
    elif args.optimizer == "rmsprop":
        return RMSProp(weights,biases,lr=args.learning_rate,beta=args.beta,epsilon=args.epsilon)
    elif args.optimizer == "adam":
        return Adam(weights,biases,lr=args.learning_rate,beta1=args.beta1,beta2=args.beta2,epsilon=args.epsilon)
    elif args.optimizer == "nadam":
        return Nadam(weights,biases,lr=args.learning_rate,beta1=args.beta1,beta2=args.beta2,epsilon=args.epsilon)
    else:
        raise ValueError("Invalid optimizer name")