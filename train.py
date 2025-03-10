import numpy as np

import argparse
from model import NeuralNetwork
from optimizers import create_optimizer
from keras.datasets import fashion_mnist, mnist
from sklearn.model_selection import train_test_split
import wandb

def create_parser():
    parser = argparse.ArgumentParser(description='Accept Command line arguments as mentioned in Code Specifications')
    
    # Added arguments as per Code Specifications of Assignment
    parser.add_argument('-wp', '--wandb_project', type=str, default='myprojectname',
                        help='Project name used to track experiments in Weights & Biases dashboard')    
    parser.add_argument('-we','--wandb_entity', type=str, default='myname',
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
    parser.add_argument('-s','--seed', type=int, default=42, help="Seed used to initialize random number generators.")
    return parser


def one_hot_encoding(labels, num_classes=10):
    Y = np.zeros((len(labels), num_classes))
    for i, j in enumerate(labels):
        Y[i,j] = 1
    return Y

def main():
    parser = create_parser()
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    config = wandb.config

    run_name = f"hl_{config.num_layers}_hs_{config.hidden_size}_bs_{config.batch_size}_ep_{config.epochs}_ac_{config.activation}_o_{config.optimizer}_lr_{config.learning_rate}_wd_{config.weight_decay}_wi_{config.weight_init}_dataset_{config.dataset}"
    wandb.run.name = run_name
    wandb.run.save()

    # Check the parsed arguments to be used in training script
    layer_sizes = [784] + [config.hidden_size]*config.num_layers + [10]
    model = NeuralNetwork(layer_sizes, config)
    
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() if config.dataset == "fashion_mnist" else mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] if config.dataset == "fashion_mnist" else ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    X_train = (train_images.astype(np.float32)-127)/128.
    Y_train = one_hot_encoding(train_labels)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)
    

    model.train(X_train, Y_train, X_val, Y_val, epochs=config.epochs)
    cor, incor = 0, 0
    for (image, label) in zip(X_train, Y_train):
        y_hat = model.forward(image.flatten())[-1]
        if label[np.argmax(y_hat)]==1:
            cor += 1
        else:
            incor += 1

    print("accuracy:", cor/(cor+incor))

if __name__ == '__main__':
    main()
