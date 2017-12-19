import dataset as ds

import autograd.numpy as np  
from autograd import grad    


train_xs, train_ys = ds.random_cat_sample('dataset/cifar-10-batches-py/data_batch_1')
train_xs2, train_ys2 = ds.random_cat_sample('dataset/cifar-10-batches-py/data_batch_3')
train_xs3, train_ys3 = ds.random_cat_sample('dataset/cifar-10-batches-py/data_batch_4')
train_xs = train_xs + train_xs2 + train_xs3
train_ys = train_ys + train_ys2 + train_ys3


test_xs,  test_ys = ds.random_cat_sample('dataset/cifar-10-batches-py/data_batch_2')

train_ys = np.reshape(train_ys, (-1,1) )


test_xs = np.transpose(test_xs)
test_ys = np.transpose(test_ys).reshape( (1,-1) )

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

def grad_descent(xs,ys,loss,net,weights):
    fw = lambda W: forward_pass(xs, net, W)
    l = lambda W: loss( fw(W), ys) 
    
    dw = grad(l)(weights)
    for i in range(0,len(weights)):
        weights[i][0] -= dw[i][0] * learning_rate
        weights[i][1] -= dw[i][1] * learning_rate
    return (l(weights), weights)

net = create_input(3072)
net = add_forward(net, 128, relu)
net = add_forward(net, 128, relu)
net = add_forward(net, 128, relu)
net = add_forward(net, 1, sigmoid)
learning_rate = 0.001

weights = init_weights(net)

batch_size = 64
chunks = int(len(train_xs) / batch_size)

def cost(yhat,y ):
    eps = 1e-18
    loss = -(y * np.log(yhat + eps) + (1-y) * np.log(1-yhat + eps))
    m = yhat.shape[1]
    cost = np.squeeze(np.mean(loss,axis=1))
    return cost

for epoch in range(0,1000):
    
    losses = []
    for item in np.array_split(list(zip(train_xs, train_ys)), chunks):
        batch_xs = [ i[0] for i in item]
        batch_ys = [ i[1] for i in item]
        batch_xs = np.transpose(batch_xs)
        batch_ys = np.transpose(batch_ys).reshape( (1,-1) )
        c, weights = grad_descent(batch_xs, batch_ys, cost, net, weights)
        losses.append(c)

    print('epoch %d is loss %f' % (epoch, np.mean(losses) ) )
    print('os loss', cost( forward_pass(test_xs, net, weights), test_ys ) )

print('loss', l(weights))
