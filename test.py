import NN as nn
import autograd.numpy as np  
import dataset as ds

train_xs, train_ys = ([],[])
for i in range(1,5):
    (xs,ys) = ds.random_cat_sample('dataset/cifar-10-batches-py/data_batch_%d' % i)
    train_xs += xs
    train_ys += ys

test_xs,  test_ys = ds.random_cat_sample('dataset/cifar-10-batches-py/data_batch_5')

train_ys = np.reshape(train_ys, (-1,1) )

test_xs = np.transpose(test_xs)
test_ys = np.transpose(test_ys).reshape( (1,-1) )

net = nn.create_input(3072)
net = nn.add_forward(net, 128, nn.relu)
net = nn.add_forward(net, 128, nn.relu)
net = nn.add_forward(net, 128, nn.relu)
net = nn.add_forward(net, 1, nn.sigmoid)
learning_rate = 0.001

weights = nn.init_weights(net)

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
        c, weights = nn.grad_descent(batch_xs, batch_ys, cost, net, weights, learning_rate = learning_rate)
        losses.append(c)

    print('epoch %d is loss %f' % (epoch, np.mean(losses) ) )
    print('os loss', cost( nn.forward_pass(test_xs, net, weights), test_ys ) )

print('loss', l(weights))
