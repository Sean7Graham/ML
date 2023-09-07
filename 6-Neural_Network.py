import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline


def plotClass(X, y, p):
    plt.figure()
    for i in range(y.shape[1]):
        if y[0, i] == 0:
            plt.plot(X[0, i], X[1, i], "r" + p)
        else:
            plt.plot(X[0, i], X[1, i], "b" + p)

    plt.show()


# Q1
num_data = 1000  # data points per class

X = np.random.uniform(-1, 1, [2, num_data])
y = X[0, :] ** 2 + X[1, :] ** 2 - 0.5 > 0
y = (y[None, :]).astype(np.int8)
plotClass(X, y, "o")
print(X.shape)
print(y.shape)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def get_loss(yhat, y):
    return np.sum(-y * np.log(yhat) - (1 - y) * np.log(1 - yhat), axis=1)


print(X.shape, y.shape)

nh1 = 10
nh2 = 5
ni = X.shape[0]
no = y.shape[0]

W1 = np.random.randn(nh1, ni)
b1 = np.random.randn(nh1, 1)

W2 = np.random.randn(nh2, nh1)
b2 = np.random.randn(nh2, 1)

W3 = np.random.randn(no, nh2)
b3 = np.random.randn(no, 1)

lr = 0.005
num_epochs = 10000
ls = []
for i in range(num_epochs):
    # forward pass
    Z1 = W1 @ X + b1
    A1 = sigmoid(Z1)

    Z2 = W2 @ A1 + b2
    A2 = sigmoid(Z2)

    Z3 = W3 @ A2 + b3
    A3 = sigmoid(Z3)

    loss = get_loss(A3, y)

    # backward pass
    dZ3 = A3 - y
    dW3 = dZ3 @ A2.T
    db3 = np.sum(dZ3, axis=1, keepdims=True)

    dZ2 = W3.T @ dZ3 * A2 * (1 - A2)
    dW2 = dZ2 @ A1.T
    db2 = np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T @ dZ2 * A1 * (1 - A1)
    dW1 = dZ1 @ X.T
    db1 = np.sum(dZ1, axis=1, keepdims=True)

    # optimization
    W3 -= lr * dW3
    b3 -= lr * db3
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    ls.append(loss)

plt.plot(ls)

ind = np.arange(X.shape[0])
np.random.shuffle(ind)

ind_val = ind[:200]
ind_train = ind[200:]

print(ind_val[:10], ind_train[:10])

Xv = X[ind_val, :]
yv = y[ind_val, :]

print(Xv.shape, yv.shape)


def get_batches(X, y, bs):
    ind = np.arange(X.shape[0])
    np.random.shuffle(ind)
    ind_start = 0
    batches = []
    ind_end = 0
    while ind_end < X.shape[0]:
        ind_end = ind_start + bs
        Xbatch = X[ind_start : min(ind_end, X.shape[0]), :]
        ybatch = y[ind_start : min(ind_end, X.shape[0]), :]
        ind_start = ind_end
        batch = (Xbatch, ybatch)
        batches.append(batch)

    return batches


batches = get_batches(X, y, 64)

for batch in batches:
    print(batch[0].shape, batch[1].shape)
