import numpy as np
from nn_lib.model import MLP

class Optimiser:
    def step(self, MLP):
        raise NotImplementedError


class SGD(Optimiser): #mini-batch  gradient descent
    def __init__(self, lr):
        self.lr = lr

    def step(self, MLP):
        for param, grad in MLP.params_and_grads():
            param -= self.lr * grad