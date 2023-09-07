import numpy as np
import matplotlib.pyplot as plt


def plotClass(X, y, p):
    plt.figure()
    for i in range(y.shape[1]):
        if y[0, i] == 0:
            plt.plot(X[0, i], X[1, i], "r" + p)
        else:
            plt.plot(X[0, i], X[1, i], "b" + p)
    plt.show()


num_data = 1000  # data points per class

X = np.random.uniform(-1, 1, [2, num_data])
y = X[0, :] ** 2 + X[1, :] ** 2 - 0.5 > 0
y = (y[None, :]).astype(np.int8)
plotClass(X, y, "o")
print(X.shape)
print(y.shape)

X = X.T
y = y.T


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# define the loss function here
# eps is a smooth parameter
def loss(yhat, y):
    eps = 1e-10
    return -np.sum(y * np.log(yhat + eps) + (1 - y) * np.log(1 - yhat + eps))


M = 3
Xbig = []
for i in range(M + 1):
    for j in range(M + 1):
        Xbig.append(X[:, 0] ** i * X[:, 1] ** j)


Xbig = np.array(Xbig).T

# w is theat
w = np.random.randn(Xbig.shape[1], 1)
lr = 0.0001

ls = []
for i in range(1000):
    # TODO: compute yhat here
    yhat = sigmoid(Xbig.dot(w))
    l = loss(yhat, y)

    # TODO: this is the gradient
    dw = Xbig.T.dot(sigmoid(Xbig.dot(w)) - y)

    # TODO: update the w here with learning rate lr
    w = w - lr * dw
    ls.append(l)

# plotClass(X.T, y.T, 'o')
plt.figure()
# plotClass(X.T, (yhat.T>0.5), 'x')

print(len(ls))
plt.figure()
plt.plot(ls)
