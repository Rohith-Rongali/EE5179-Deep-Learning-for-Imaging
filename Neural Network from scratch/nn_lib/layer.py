class layer:
    def __init__(self,name):
        self.name=name
        self.params={}
        self.grad = {}
    
    def forward(self,input):
        raise NotImplementedError
        
    def backward(self,grad):
        raise NotImplementedError
        
def Glorot_init(out_size,inp_size):
    M = np.sqrt(6/(inp_size+out_size))
    return np.random.uniform(-M,M,(out_size,inp_size))


sigmoid = lambda x: 1/(1+np.exp(-x))

def sigmoid_del(x):
    h = sigmoid(x)
    return h*(1-h)

def tanh(x):
    return np.tanh(x)

def tanh_del(x):
    y = tanh(x)
    return 1 - y ** 2

relu = lambda x: np.maximum(0, x)

def relu_del(x):
    y = relu(x)
    der = np.zeros(y.shape)
    der[np.where(y > 0)] = 1
    return der

affine = lambda x: x

def affine_del(x):
    return np.ones(x.shape)



    
class Activation(layer):
    def __init__(self,name,inp_size,out_size,f,f_del):
        super().__init__(name)
        self.f = f
        self.f_del = f_del
        self.params["w"] = Glorot_init(inp_size,out_size)
        self.params["b"] = np.zeros(out_size)
        self.grad["w"] = np.zeros((inp_size,out_size))
        self.grad["b"] = np.zeros(out_size)
        
    def forward(self,inputs):
        self.inputs = inputs
        return self.f( inputs @ self.params['w'] + self.params["b"])
    
    def backward(self,grad):
        self.grad["w"] = self.inputs.T @ grad
        self.grad["b"] = np.sum(grad, axis=0)
        return (grad @ self.params['w'].T)*self.f_del(self.inputs)
    
    
class Sigmoid(Activation):
    def __init__(self,name,inp_size,out_size):
        super().__init__( name,inp_size,out_size,sigmoid, sigmoid_del)
        
class Tanh(Activation):
    def __init__(self,name,inp_size,out_size):
        super().__init__(name,inp_size,out_size,f = tanh, f_del = tanh_del)
        
class Relu(Activation):
    def __init__(self,name,inp_size,out_size):
        super().__init__(name,inp_size,out_size,f = relu,f_del = relu_del)

class Affine(Activation):
    def __init__(self,name,inp_size,out_size):
        super().__init__(name,inp_size,out_size,f = affine,f_del = affine_del)
