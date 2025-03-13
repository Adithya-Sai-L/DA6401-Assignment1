# DA6401-Assignment1
## GOAL
Implement and use gradient descent (and its variants) and get familiar with Wandb

## Problem Statement
In this assignment you need to implement a feedforward neural network and write the backpropagation code for training the network. We strongly recommend using numpy for all matrix/vector operations. You are not allowed to use any automatic differentiation packages. This network will be trained and tested using the Fashion-MNIST dataset. Specifically, given an input image (28 x 28 = 784 pixels) from the Fashion-MNIST dataset, the network will be trained to classify the image into 1 of 10 classes.

# Question 1 (2 Marks)

['plot_images.py'](plot_images.py)
**Usage** python plot_images.py -wp wandb_project -we wandb_entity -d dataset -s seed