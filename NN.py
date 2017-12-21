import autograd.numpy as np  
from autograd import grad    

def create_input(size):
    return [{'dim':size, 'act': identity}]

def add_forward(layers, size, activation):
    layers.append({'dim':size, 'act':activation})
    return layers

def identity(z):
    return z

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z) )

def tanh(z):
    return np.tanh(z)

def relu(z):
    return np.maximum(0, z)

def compute_activation(act, A, W, b):
    return act( np.dot(W,A) + b )

def init_weights(layers):
    weights = []
    for l in range(1,len(layers)):
        W = np.random.randn( layers[l]['dim'], layers[l-1]['dim'] ) * 0.01
        b = np.zeros( (layers[l]['dim'], 1) )
        weights.append( [W,b] )
    return weights

def forward_pass(X, layers, weights):
    A_prev = X
    for (l,[W,b]) in zip(layers[1:], weights):
        A = compute_activation(l['act'], A_prev, W, b)
        A_prev = A
    return A_prev

def grad_descent(xs,ys,loss,net,weights, learning_rate=0.001):
    fw = lambda W: forward_pass(xs, net, W)
    l = lambda W: loss( fw(W), ys) 
    
    dw = grad(l)(weights)
    for i in range(0,len(weights)):
        weights[i][0] -= dw[i][0] * learning_rate
        weights[i][1] -= dw[i][1] * learning_rate
    return (l(weights), weights)

