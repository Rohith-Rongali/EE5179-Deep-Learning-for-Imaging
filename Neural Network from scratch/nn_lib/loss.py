import numpy as np

        
class CE_loss:
    def __call__(self, pred, actual):
        ce_loss = -actual*np.log(pred)
        return np.mean(ce_loss)
                    
    def backward(self,pred,actual):
        m = actual.shape[0]
        grad = (pred - actual)/m
        return grad