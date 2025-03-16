import numpy as np

import argparse
from model import NeuralNetwork
from optimizers import create_optimizer
from keras.datasets import fashion_mnist, mnist

def create_parser():
    parser = argparse.ArgumentParser(description='Accept Command line arguments as mentioned in Code Specifications')
    
    # Added arguments as per Code Specifications of Assignment
    parser.add_argument('-d','--dataset', type=str, default='fashion_mnist', choices=["mnist", "fashion_mnist"], help="Choose dataset from ['mnist', 'fashion_mnist']")
    # parser.add_argument('-e','--epochs', type=int, default=1, help="Number of epochs to train neural network.")
    # parser.add_argument('-b','--batch_size', type=int, default=4, help='Batch size used to train neural network.')
    parser.add_argument('-l','--loss', type=str, default='cross_entropy', choices=["mean_squared_error", "cross_entropy"], help="Choose loss from ['mean_squarred_error', 'cross_entropy']")                        
    parser.add_argument('-nhl','--num_layers', type=int, default=1, help="Number of hidden layers used in feedforward neural network.")
    parser.add_argument('-sz','--hidden_size', type=int, default=4, help="Number of hidden neurons in a feedforward layer.")
    parser.add_argument('-a','--activation', type=str, default="sigmoid", choices=["identity", "sigmoid", "tanh", "ReLU"], help='Choose activation from ["identity", "sigmoid", "tanh", "ReLU"]')    
    parser.add_argument('-w_i','--weight_init', type=str, default="random", choices=["random", "Xavier"], help="Choose initialization from ['random', 'Xavier']")
    parser.add_argument('-b','--batch_size', type=int, default=4, help='Batch size used to train neural network.')
    parser.add_argument('-lr','--learning_rate', type=float, default=0.1, help="Learning rate used to optimize model parameters")
    parser.add_argument('-o','--optimizer', type=str, default="sgd", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], help='Choose optimizer from ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]')
    parser.add_argument('-w_d','--weight_decay', type=float, default=.0, help="Weight decay used by optimizers.")
    parser.add_argument('-w', '--weights', type=str, required=True, help='Path to the weights file')
    return parser


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
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] if args.dataset == "fashion_mnist" else ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    params = np.load(args.weights, allow_pickle=True)
    
    model.weights = params.item().get('weights')
    model.biases = params.item().get('biases')
    # print(model.weights)
    X_test = (test_images.astype(np.float32)-127)/128.
    Y_test = one_hot_encoding(test_labels)

    cor, incor = 0, 0
    for (image, label) in zip(X_test, Y_test):
        y_hat = model.forward(image.flatten())[-1]
        if label[np.argmax(y_hat)]==1:
            cor += 1
        else:
            incor += 1


    print("Final Test accuracy:", cor/(cor+incor))

if __name__ == '__main__':
    main()
