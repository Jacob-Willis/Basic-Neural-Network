# Imports
import NeuralNetwork as nn
import numpy as np 

with np.load('mnist.npz') as data:
    training_images = data['training_images']
    training_labels = data['training_labels']

"""
Initialises the layer sizes for the neural network.  First value is inputs,
last in outputs and any number in-between are the 'hidden' layers and how
many neurons they have
"""

layer_sizes = (784, 16, 16, 10)

"""
Number of images used for training the A.I.  The rest rest of the images are
used for testing the performance
"""

training_set_size = 5000

training_set_images = training_images[:training_set_size]
training_set_labels = training_labels[:training_set_size]

test_set_images = training_images[training_set_size:]
test_set_labels = training_labels[training_set_size:]

# Calls the __init__ method and sets 'net' as the network
net = nn.NeuralNetwork(layer_sizes)

# Evalualtes the performance of the system without any training sessions
net.print_accuracy(training_images, training_labels)
net.calculate_average_cost(test_set_images, test_set_labels)

# First training sessions
net.train_sgd(training_set_images, training_set_labels, 8, 20, 4.0)

# Evalualtes the performance of the system without any training sessions
net.print_accuracy(training_images, training_labels)
net.calculate_average_cost(test_set_images, test_set_labels)

# Second training session
net.train_sgd(training_set_images, training_set_labels, 8, 20, 2.0)

# Evaluate performance after second training session
net.print_accuracy(test_set_images, test_set_labels)
net.calculate_average_cost(test_set_images, test_set_labels)
