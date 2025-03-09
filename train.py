import numpy as np

import argparse
from activations import ACTIVATIONS
from losses import LOSSES
from optimizers import create_optimizer

from keras.datasets import fashion_mnist, mnist

def create_parser():
    parser = argparse.ArgumentParser(description='Accept Command line arguments as mentioned in Code Specifications')
    
    # Added arguments as per Code Specifications of Assignment
    parser.add_argument('-wp', '--wandb_project', type=str, default='myprojectname',
                        help='Project name used to track experiments in Weights & Biases dashboard')
    
    parser.add_argument('-we','--wandb_entitiy', type=str, default='myname',
                        help='Wandb Entity used to track experiments in the Weights & Biases dashboard.')
    
    parser.add_argument('-d','--dataset', type=str, default='fashion_mnist', choices=["mnist", "fashion_mnist"], help="Choose dataset from ['mnist', 'fashion_mnist']")

    parser.add_argument('-e','--epochs', type=int, default=1, help="Number of epochs to train neural network.")

    parser.add_argument('-b','--batch_size', type=int, default=4, help='Batch size used to train neural network.')

    parser.add_argument('-l','--loss', type=str, default='cross_entropy', choices=["mean_squared_error", "cross_entropy"], help="Choose loss from ['mean_squarred_error', 'cross_entropy']")
                        
    parser.add_argument('-o','--optimizer', type=str, default="sgd", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], help='Choose optimizer from ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]')
                        
    parser.add_argument('-lr','--learning_rate', type=float, default=0.1, help="Learning rate used to optimize model parameters")

    parser.add_argument('-m','--momentum', type=float, default=  0.5, help="Momentum used by momentum and nag optimizers.")

    parser.add_argument('-beta','--beta', type=float, default=0.5, help="Beta used by rmsprop optimizer")

    parser.add_argument('-beta1','--beta1', type=float, default=0.5, help="Beta1 used by adam and nadam optimizers.")

    parser.add_argument('-beta2','--beta2', type=float, default=0.5, help="Beta2 used by adam and nadam optimizers.")

    parser.add_argument('-eps','--epsilon', type=float, default=0.000001, help="Epsilon used by optimizers.")

    parser.add_argument('-w_d','--weight_decay', type=float, default=.0, help="Weight decay used by optimizers.")

    parser.add_argument('-w_i','--weight_init', type=str, default="random", choices=["random", "Xavier"], help="Choose initialization from ['random', 'Xavier']")

    parser.add_argument('-nhl','--num_layers', type=int, default=1, help="Number of hidden layers used in feedforward neural network.")

    parser.add_argument('-sz','--hidden_size', type=int, default=4, help="Number of hidden neurons in a feedforward layer.")

    parser.add_argument('-a','--activation', type=str, default="sigmoid", choices=["identity", "sigmoid", "tanh", "ReLU"], help='Choose activation from ["identity", "sigmoid", "tanh", "ReLU"]')
    
    return parser


class NeuralNetwork:
    def __init__(self, layer_sizes, args=None):
        self.layer_sizes = layer_sizes
        self.weights = [np.random.randn(y, x) * np.sqrt(2/x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.zeros((y,)) for y in layer_sizes[1:]]
        self.activation = ACTIVATIONS[args.activation]
        self.softmax = ACTIVATIONS["Softmax"]
        self.loss = LOSSES[args.loss]()
        self.optimizer = create_optimizer(args, self.weights, self.biases)

    def forward(self, X):
        h_s = [X]
        for W, b in zip(self.weights, self.biases):
            a = np.dot(W, h_s[-1]) + b
            if W is self.weights[-1]:
                h = self.softmax.forward(a)
            else:
                h = self.activation.forward(a)
            h_s.append(h)
        return h_s
    
    def calc_loss(self, y_hat, y):
        return self.loss.forward(y_hat, y)
    
    def backward(self, h_s, y):
        dH = self.loss.backward(h_s[-1], y)


        dW = [np.zeros((y, x)) for x, y in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
        dB = [np.zeros((y,)) for y in self.layer_sizes[1:]]

        for i in reversed(range(1,len(self.layer_sizes))):
            if i == len(self.layer_sizes)-1:
                dA = np.dot(self.softmax.backward(h_s[i]), dH)
            else:
                dA = np.dot(self.activation.backward(h_s[i]), dH)
            dW[i-1] = np.dot(np.expand_dims(dA, axis=1), np.expand_dims(h_s[i-1], axis=0))
            dB[i-1] = dA
            dH = np.dot(self.weights[i-1].T, dA)

        return dW, dB
    
    def train(self, train_images, Y_train, epochs=1):
        for i in range(epochs):
            total_loss = 0
            for (image, label) in zip(train_images, Y_train):
                h_s = self.forward(image.flatten())
                y_hat = h_s[-1]
                total_loss += self.calc_loss(y_hat, label)
                dW, dB = self.backward(h_s, label)
                self.optimizer.update(dW, dB)
            print(f"Epoch{i} train loss: {total_loss/len(train_images)}")
            total_loss = 0
            for (image, label) in zip(train_images, Y_train):
                h_s = self.forward(image.flatten())
                y_hat = h_s[-1]
                total_loss += self.calc_loss(y_hat, label)
            print(f"Epoch{i} test loss: {total_loss/len(train_images)}")

def one_hot_encoding(labels, num_classes=10):
    Y = np.zeros((len(labels), num_classes))
    for i, j in enumerate(labels):
        Y[i,j] = 1
    return Y

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    # Check the parsed arguments to be used in training script
    layer_sizes = [784] + [args.hidden_size]*args.num_layers + [10]
    model = NeuralNetwork(layer_sizes, args)
    
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() if args.dataset == "fashion_mnist" else mnist.load_data()
    train_images.shape, train_labels.shape, test_images.shape, test_labels.shape
    # TODO: Add Training Code

    X_train = (train_images.astype(np.float32)-127)/128.
    Y_train = one_hot_encoding(train_labels)

    model.train(X_train, Y_train, epochs=5)

    cor, incor = 0, 0
    for (image, label) in zip(train_images, Y_train):
        y_hat = model.forward(image.flatten())[-1]
        if label[np.argmax(y_hat)]==1:
            cor += 1
        else:
            incor += 1

    print("accuracy:", cor/(cor+incor))

if __name__ == '__main__':
    main()
