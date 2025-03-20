# DA6401-Assignment1
## GOAL
Implement and use gradient descent (and its variants) and get familiar with Wandb

## Problem Statement
In this assignment you need to implement a feedforward neural network and write the backpropagation code for training the network. We strongly recommend using numpy for all matrix/vector operations. You are not allowed to use any automatic differentiation packages. This network will be trained and tested using the Fashion-MNIST dataset. Specifically, given an input image (28 x 28 = 784 pixels) from the Fashion-MNIST dataset, the network will be trained to classify the image into 1 of 10 classes.

# WandB Report

[Link to WandB Report](https://wandb.ai/ns25z041-iit-madras/Programming_Assignment_1/reports/Adithya-Sai-Lenka-s-DA6401-Assignment-1--VmlldzoxMTc3MzIyNg?accessToken=mfov88bif27x6fengkks7zodl5sw2udlbk3jlugo3neyqh0lu63im6oh75oqljyp)

# Github Repository

[Link to Github Repository](https://github.com/Adithya-Sai-L/DA6401-Assignment1)

## File Structure

Below is an overview of the repository structure, which is organized to enhance modularity and ease of navigation:


---
```bash
├── ckpts/                     
# Directory for storing saved model files corresponding to the best performing configurations.
├── activations.py           
# Contains all activation functions (Softmax, sigmoid, tanh, ReLU, identity) and their derivatives.
├── losses.py                 
# Contains loss functions (cross-entropy and mean squared error) with their gradient implementations.
├── model.py                
# Implements the NeuralNetwork class, which includes methods for the forward and backward passes.
├── optimizers.py           
# Contains implementations of different optimization algorithms (SGD, momentum, NAG, RMSprop, Adam, Nadam).
└── plot_images.py                
# Visualizing sample images from the Fashion-MNIST or MNIST dataset.
├── sweep.py                      
# python code to run WandB sweeps with configs defined
├── train.py                    
# Main training script that orchestrates data loading, model training, logging, and model saving.
├── test.py
# Script to test model on Fashion-MNIST or MNIST dataset.
```


# Create Environment

conda create -n dl_pa1 dl_pa1.yaml

# Question 1 (2 Marks)

[plot_images.py](plot_images.py) \
**Usage:** \
```python plot_images.py -wp wandb_project -we wandb_entity -d dataset -s seed```

# Question 2

[train.py](train.py) takes arguments:
- -nhl &emsp;&nbsp; for number of hidden layers
- -sz  &emsp;&emsp;for hidden layer size

# Question 3

[optimizers.py](optimizers.py) implements the following optimizers:
- sgd
- momentum based gradient descent
- nesterov accelerated gradient descent
- rmsprop
- adam
- nadam

Note: Nesterov momentum not defined exactly as per algorithm since it requires maintaining lookahead weights and computing gradients w.r.t the lookahead weights. Similar update for NAdam optimizer.

New optimization algorithm Eve can be implemented by
1. creating class Eve extending Base optimizer in [optimizer.py](optimizers.py)
2. updating create_optimizer function to return Eve optimizer when args.optimizer == 'eve'
3. modify parser in [train.py](train.py) adding 'eve' to choices of optimizer argument.


[train.py](train.py) supports 
- -b for batch_size, with logic handled in [model.py](model.py)

# Question 4

Question 8 requires to run the experiments with Mean Squared Error Loss. $\mathcal{L} = \frac{1}{K}\sum_{c=1}^{K}(\hat{y}-y)^2$

$$\frac{\partial \mathcal{L}}{\partial \hat{y}}$$ 
is a dense vector with many non-zero values across the output dimension as opposed Cross Entropy loss for categorical distribution, which is sparse with only the label corresponding to groundtruth having non-zero value.

The derivative of $\hat{y}$ w.r.t pre-activation $a_L$ is a Jacobian matrix since both $\hat{y}$ and $a_L$ are vectors.$$\hat{y} = Softmax(a_L)$$ Since softmax is not an element-wise function, it's derivative is 2-D matrix.

Hence the derivatives of all activation functions in [activations.py](activations.py) are Jacobian matrices. Since the other activations implemented (ReLU, tanh, sigmoid, and identity) are element-wise functions, their derivatives are diagonal matrices. For all the above functions, including Softmax, the derivative can be computed with only the activations computed during the forward pass and do not require the pre-activations.

Additional loss functions can be implemented in [losses.py](losses.py).
Additional activation functions can be implemented along with their derivatives in [activations.py](activations.py)


Run sweep by running

**Usage:** \
```python sweep.py -wp wandb_project -we wandb_entity -d dataset```

WandB does not have conditional sweep. It would be useful when certain hyperparameters depend on the value of other hyperparameters. For example, momentum hyperparameter does not affect training when the optimizer is sgd, and beta1 and beta2 hyperparameters have an effect only when the training utilizes adam or nadam optimizer and not rmsprop.

WandB provides 3 options for hyperparameter search strategies:
- grid
- random
- Bayesian search

Since grid search iterates over every combination of hyperparameter values, it is not suitable for this assignment due to the exponential number of combinations in the number of hyperparameters.

Random search chooses a random, uniformed set of hyperparameter values based on a distribution, which may be constant, categorical, uniform, log_uniform, etc. 

Bayesian search makes informed decisions as opposed to grid and random, by using a probablistic model to decide which values to use through an iterative process of testing values on a surrogate function before evaluating the objective function.

I used the bayesian search strategy to maximize the best_validation_accuracy of the model across its epochs.

# Question 6

Observations:

- According to the correlation plot with respect to the best validation accuracy, the most important parameter is weight_decay which has a negative correlation. This is probably because of the possible values of weight_decay being 0, 0.0005 and 0.5. When weight_decay is 0.5, the model suffers from excessive regularization and does not seem to perform as well on even the training data (suffering from high bias).

- The next most important parameter is num_layers. As expected, when the number of layers in the model increases, its complexity and learning capacity increases, and it is able to achieve lower loss and higher accuracy.

- Despite the models achieving pretty good accuracy, there is still a difference of 3-4% between the training and validation accuracy, implying there is a slight over-fitting problem, and need for more regularization.


To achieve a higher accuracy like around 95%, the complexity of the model needs to be increased. Further, to reduce the gap between training and validation accuracies, there needs to be sufficient regularization for the model to generalize well. Since  the input in this task consists of images, the model needs to be taught of the translational invariance of the input. For this case, the inductive bias provided by convolutional operator may be useful to push the performance of the model to beyond 95%.

# Question 7

[test.py](test.py) \
**Usage:** \
```python test.py -w path_to_weights.npy -d dataset -nhl num_layers -sz hidden_size -a activation ```

Will create confusion matrix and log on wandb

# Question 8

My sweep config included mean_squarred_error as a hyperparameter. Hence another sweep was conducted and the correlation between the choice of loss function and best_validation_accuracy was reversed. Therefore, in my experiments, there does NOT seem to be a clear winner for the choice of loss function. Since the target is a categorical distribution, the output activation being fixed to softmax seems to be sufficient to train the network on both mean_squarred_error and cross_entropy losses.

# Question 9

[Link to Github Repository](https://github.com/Adithya-Sai-L/DA6401-Assignment1)

# Question 10

| Configuration # | Epochs | Batch Size | Loss Function | Learning Rate | Optimizer | Momentum | Weight Decay | Weight Init | Num Layers | Hidden Size | Activation | Training Accuracy | Best Validation Accuracy | Test Accuracy |
|-----------------|--------|------------|---------------|---------------|-----------|----------|-------------|------------|------------|-------------|------------|-------------------|--------------------------|--------------|
| 1               | 10     | 16         | cross_entropy | 0.0001        | momentum  | 0.9      | 0.0005      | Xavier     | 3          | 128        | ReLU       | 98.21%           | 96.75%                  | 97.04%     |
| 2               | 10     | 32         | mean_squared_error | 0.0001    | nag       | 0.9      | 0.0005      | Xavier     | 4          | 128        | tanh       | 98.10%           | 96.93%                  | 96.93%     |
| 3               | 10     | 64         | cross_entropy | 0.0001        | nag       | 0.9      | 0.0         | Xavier     | 5          | 128        | ReLU       | 97.38%           | 96.37%                  | 96.13%     |

