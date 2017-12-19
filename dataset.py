import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_cats(filename):
    dat = unpickle(filename)
    # for k in dat:
    #     print('key',k)
    # raise 'poes'
    labels = dat[b'labels']
    data = dat[b'data']

    cats = []
    other = []

    for i in range(0, len(data)):
        if labels[i] == 3:
            cats.append(data[i])
        else:
            other.append(data[i])
    other = other[:len(cats)]

    return (cats, other)

def random_cat_sample(filename):
    cats, not_cats = load_cats(filename)
    
    cats = list(zip(cats, [1]*len(cats)))
    not_cats = list(zip(not_cats, [0]*len(not_cats)))
    
    np.random.shuffle(cats)
    np.random.shuffle(not_cats)
    
    train = cats[0:500] + not_cats[0:500]
    np.random.shuffle(train)
    train_xs = [ item[0] for item in train ]
    train_ys = [ item[1] for item in train ]

    return (train_xs, train_ys)


