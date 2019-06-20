# Imports
import NeuralNetwork as nn
import numpy as np 

# Initialises the layer sizes for the neural network.  First value is inputs,
# last in outputs and any number in-between are the 'hidden' layers and how
# many neurons they have
layer_sizes = (3,5,10) 
x = np.ones((layer_sizes[0],1)) 

# Calls the __init__ method and sets 'net' as the network
net = nn.NeuralNetwork(layer_sizes) 
# Calculates the prediction the network gives
prediction = net.predict(x) 

print(prediction)