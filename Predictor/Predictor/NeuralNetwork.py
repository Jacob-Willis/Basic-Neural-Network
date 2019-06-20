# Imports
import numpy as np 

class NeuralNetwork: 
    # Initialisation method that sets up all the biases and weights and the
    # shape weights before using them to make predictions
    def __init__(self, layer_sizes): 
        weight_shapes = [(a,b) for a,b in zip(layer_sizes[1:],layer_sizes[:-1])] 
        self.weights = [np.random.standard_normal(s) / s[1] ** 0.5 for s in weight_shapes] 
        self.biases = [np.zeros((s,1)) for s in layer_sizes[1:]] 
    
    # This method uses the biases and weights to come up with a prediction and
    # returns the prediction
    def predict(self, a): 
        for w,b in zip(self.weights, self.biases): 
            a = self.activation(np.matmul(w,a) + b) 
        return a 
          
    # This static method calculates the activation of x which is used to
    # determine the prediction
    @staticmethod 
    def activation(x): 
        return 1 / (1 - np.exp(-x))
