import numpy as np
from activations import ACTIVATIONS
from losses import LOSSES
from optimizers import create_optimizer
import wandb

class NeuralNetwork:
    def __init__(self, layer_sizes, args=None):
        self.layer_sizes = layer_sizes
        self.weights = [np.random.randn(y, x) * np.sqrt(2/x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])] if args.weight_init == "random" else [np.random.uniform(-np.sqrt(6/(x+y)), np.sqrt(6/(x+y)), (y, x)) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.zeros((y,)) for y in layer_sizes[1:]]
        self.activation = ACTIVATIONS[args.activation]
        self.softmax = ACTIVATIONS["Softmax"]
        self.loss = LOSSES[args.loss]()
        self.optimizer = create_optimizer(args, self.weights, self.biases)
        self.batch_sz = args.batch_size
        self.args = args
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
    
    def train(self, train_images, Y_train, val_images, Y_val, epochs=1):
        # old_weights, old_biases = self.weights, self.biases
        datapoints_seen = 0
        dW, dB = [np.zeros_like(W) for W in self.weights], [np.zeros_like(B) for B in self.biases]
        best_val_accuracy = 0
        for i in range(epochs):
            train_correct, train_incorrect, val_correct, val_incorrect = 0, 0, 0, 0
            total_loss = 0
            for (image, label) in zip(train_images, Y_train):
                datapoints_seen+=1
                h_s = self.forward(image.flatten())
                y_hat = h_s[-1]
                total_loss += self.calc_loss(y_hat, label)
                dW_dp, dB_dp = self.backward(h_s, label)
                dW = [dW[layer] + dW_dp[layer] for layer in range(len(self.weights))]
                dB = [dB[layer] + dB_dp[layer] for layer in range(len(self.biases))]

                if label[np.argmax(y_hat)]==1:
                    train_correct += 1
                else:
                    train_incorrect += 1

                if datapoints_seen % self.batch_sz == 0:
                    self.optimizer.update(dW, dB)
                    dW, dB = [np.zeros_like(W) for W in self.weights], [np.zeros_like(B) for B in self.biases]

            print(f"Epoch{i} train loss: {total_loss/len(train_images)}")


            val_loss = 0
            for (image, label) in zip(val_images, Y_val):
                h_s = self.forward(image.flatten())
                y_hat = h_s[-1]
                val_loss += self.calc_loss(y_hat, label)
                if label[np.argmax(y_hat)]==1:
                    val_correct += 1
                else:
                    val_incorrect += 1
            val_accuracy = val_correct/(val_correct+val_incorrect)
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                params = {"weights": self.weights, "biases": self.biases}
                np.save(f"ckpts/{self.args.dataset}/hl_{self.args.num_layers}_hs_{self.args.hidden_size}_bs_{self.args.batch_size}_ep_{self.args.epochs}_ac_{self.args.activation}_o_{self.args.optimizer}_lr_{self.args.learning_rate}_wd_{self.args.weight_decay}_wi_{self.args.weight_init}_loss_{self.args.loss}.npy", params)
            wandb.log({"train_loss": total_loss/len(train_images), "val_loss": val_loss/len(val_images), "train_accuracy": train_correct/(train_correct+train_incorrect), "val_accuracy": val_correct/(val_correct+val_incorrect)})
            print(f"Epoch{i} val loss: {val_loss/len(val_images)}")
        wandb.log({"best_val_accuracy": best_val_accuracy})
