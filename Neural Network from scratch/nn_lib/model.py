from nn_lib.layer import layer
import numpy as np

class MLP:
    def __init__(self,layers):
        self.layers = layers
        
    def forward(self,inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def backward(self,grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def params_and_grads(self):
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grad[name]
                yield param, grad
