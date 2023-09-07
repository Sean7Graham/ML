# Linear Regression for general functions
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import math

pi = math.pi

data_set_size = 20
noise_level = 0.3

# generate 20 numbers from -1 to 1 with equal stepsize
x = np.linspace(-1, 1, data_set_size)

# generate training target (noise contaminated!)
y = np.sin(2 * pi * 0.5 * x) + noise_level * np.random.randn(x.size)

# plot the training data points
plt.plot(x, y, "ro")

# plot the true function
plt.plot(np.linspace(-1, 1, 50), np.sin(2 * pi * 0.5 * np.linspace(-1, 1, 50)), "black")

N = x.shape[0]
M = 10
X = []
for i in range(M + 1):
    X.append(x**i)

X = np.array(X).T

print("x", X.shape)
print("y", y.shape)

# It useful to first write a function that calculates the loss we will use this to measure our progress


def MSE_loss(X, Y, theta):
    N = X.shape[0]
    mean_loss = 1 / N * np.sum(np.square(X @ theta - Y))
    return mean_loss


# a quick sanity check for the loss function, If this cell throws and error your loss function still needs work!
assert MSE_loss(np.array([0]), np.array([0]), np.array([0])) == 0
assert MSE_loss(np.array([1]), np.array([1]), np.array([1])) == 0
assert MSE_loss(np.array([1]), np.array([1]), np.array([0])) == 1
assert MSE_loss(np.array([1]), np.array([0]), np.array([1])) == 1


# It is also useful to have a function that calculate gradient at a given point
def grad(X, Y, theta):
    n = X.shape[0]
    gradient = 2 / n * (X.T @ X @ theta - X.T @ Y)
    return gradient


# a quick sanity check for your gradient function, If this cell throws and error your gradient fucntion still needs work!
assert la.norm(grad(np.array([[0, 0], [0, 0]]), np.zeros((2, 1)), np.ones((2, 1)))) == 0
assert la.norm(grad(np.array([[1, 0], [0, 0]]), np.zeros((2, 1)), np.ones((2, 1)))) == 1
assert (
    la.norm(grad(np.array([[4, 3], [4, 3]]), np.zeros((2, 1)), np.ones((2, 1)))) == 70
)

# In order to use an iterative optimisation method we need an initial guess:

theta = np.zeros(M + 1)

gd_losses = []
gd_losses.append(MSE_loss(X, y, theta))

# ... and a step size

eta = 0.5


number_of_iteratons = 1000

for iter in range(number_of_iteratons):
    # ----------------------------
    # Your code here
    theta = theta - eta * grad(X, y, theta)

    # ----------------------------
    loss = MSE_loss(X, y, theta)
    gd_losses.append(loss)
    if iter % 100 == 99:
        print(loss)

plt.plot(np.arange(len(gd_losses)), gd_losses, "red")


def check_armijo_goldstein_condtions(grad, loss, trail_point_loss, eta, c):
    grad_norm = np.linalg.norm(grad)
    if trail_point_loss <= loss - c * eta * grad_norm:
        return True
    else:
        return False


theta = np.zeros(M + 1)


line_search_losses = []
line_search_losses.append(MSE_loss(X, y, theta))

c = 0.001
gamma = 10
alpha = 0.8
max_k = 50  # make number of line search steps

number_of_iteratons = 1000
loss = MSE_loss(X, y, theta)

eta = gamma


for iter in range(number_of_iteratons):
    for k in range(max_k):
        theta_dash = theta - eta * grad(X, y, theta)
        trial_point_loss = MSE_loss(X, y, theta_dash)

        if (
            check_armijo_goldstein_condtions(
                grad(X, y, theta), loss, trial_point_loss, eta, c
            )
            == True
        ):
            theta = theta_dash
            break
        else:
            eta = eta * alpha

    # once we have found a point that satisfies the condition we take a step using this step size
    # ----------------------------
    eta = gamma  # set eta back to its intial value

    loss = MSE_loss(X, y, theta)

    line_search_losses.append(loss)

    if iter % 100 == 99:
        print(loss)


plt.plot(np.arange(len(gd_losses)), gd_losses, "red")
plt.plot(np.arange(len(line_search_losses)), line_search_losses, "green")

theta = np.zeros(M + 1)
momentum = np.zeros(M + 1)

gd_with_momenutm_losses = []
gd_with_momenutm_losses.append(MSE_loss(X, y, theta))

# ... and a step size

eta = 1e-2
mu = 0.9
m = 0

number_of_iteratons = 1000

for iter in range(number_of_iteratons):
    # ----------------------------
    # Your code here
    m = mu * m - eta * grad(X, y, theta)
    theta = theta + m

    # ----------------------------
    loss = MSE_loss(X, y, theta)
    gd_with_momenutm_losses.append(loss)
    if iter % 100 == 99:
        print(loss)

plt.plot(np.arange(len(gd_losses)), gd_losses, "red", label="gd")
plt.plot(np.arange(len(line_search_losses)), line_search_losses, "green")
plt.plot(
    np.arange(len(gd_with_momenutm_losses)),
    gd_with_momenutm_losses,
    "blue",
    label="gd_with_momentum",
)

theta = np.zeros(M + 1)
momentum = np.zeros(M + 1)

gd_with_nesterov_momenutm = []
gd_with_nesterov_momenutm.append(MSE_loss(X, y, theta))

# ... and a step size

eta = 1e-2
mu = 0.9
m = 0

number_of_iteratons = 1000

for iter in range(number_of_iteratons):
    # ----------------------------
    # Your code here
    m = mu * m - eta * grad(X, y, theta)
    theta = theta - eta * grad(X, y, theta) + mu * m
    # ----------------------------
    loss = MSE_loss(X, y, theta)
    gd_with_nesterov_momenutm.append(loss)
    if iter % 100 == 99:
        print(loss)

plt.plot(np.arange(len(gd_losses)), gd_losses, "red", label="gd")
plt.plot(np.arange(len(line_search_losses)), line_search_losses, "green")
plt.plot(
    np.arange(len(gd_with_momenutm_losses)),
    gd_with_momenutm_losses,
    "blue",
    label="gd_with_momentum",
)
plt.plot(
    np.arange(len(gd_with_nesterov_momenutm)),
    gd_with_nesterov_momenutm,
    "yellow",
    label="gd_with_momentum",
)

theta = np.zeros(M + 1)

second_order_losses = []
second_order_losses.append(MSE_loss(X, y, theta))

# ... note here we don't need learning rate

number_of_iteratons = 1000
hessian = 2 / X.shape[0] * (X.T @ X)

for iter in range(number_of_iteratons):
    # ----------------------------
    # Your code here
    theta = theta - np.linalg.inv(hessian) @ grad(X, y, theta)

    # ----------------------------
    loss = MSE_loss(X, y, theta)
    second_order_losses.append(loss)
    if iter % 100 == 99:
        print(loss)

plt.plot(np.arange(len(second_order_losses)), second_order_losses, "black")
plt.plot(np.arange(len(gd_losses)), gd_losses, "red", label="gd")
plt.plot(np.arange(len(line_search_losses)), line_search_losses, "green")
plt.plot(
    np.arange(len(gd_with_momenutm_losses)),
    gd_with_momenutm_losses,
    "blue",
    label="gd_with_momentum",
)
plt.plot(
    np.arange(len(gd_with_nesterov_momenutm)),
    gd_with_nesterov_momenutm,
    "yellow",
    label="gd_with_momentum",
)

import random

theta = np.zeros(M + 1)

coord_dec_losses = []
coord_dec_losses.append(MSE_loss(X, y, theta))

number_of_iteratons = 1000
eta = 0.1

for iter in range(number_of_iteratons):
    # ----------------------------
    # Your code here
    p = np.identity(M + 1)[:, random.randint(0, M)]
    selections = np.array(
        [
            MSE_loss(X, y, theta - eta * p),
            MSE_loss(X, y, theta),
            MSE_loss(X, y, theta + eta * p),
        ]
    )
    min_selection = np.min(selections)
    index = np.argmin(selections)
    if index == 0:
        theta -= eta * p
    elif index == 2:
        theta += eta * p
    # --------------------------
    loss = MSE_loss(X, y, theta)
    coord_dec_losses.append(loss)
    eta *= 0.999
    if iter % 100 == 99:
        print(loss)

plt.plot(np.arange(len(second_order_losses)), second_order_losses, "black")
plt.plot(np.arange(len(gd_losses)), gd_losses, "red", label="gd")
plt.plot(np.arange(len(line_search_losses)), line_search_losses, "green")
plt.plot(
    np.arange(len(gd_with_momenutm_losses)),
    gd_with_momenutm_losses,
    "blue",
    label="gd_with_momentum",
)
plt.plot(
    np.arange(len(gd_with_nesterov_momenutm)),
    gd_with_nesterov_momenutm,
    "yellow",
    label="gd_with_momentum",
)
plt.plot(np.arange(len(coord_dec_losses)), coord_dec_losses, "purple")

theta = np.zeros(M + 1)

grad_free_losses = []
grad_free_losses.append(MSE_loss(X, y, theta))

number_of_iteratons = 1000
eta = 0.1

for iter in range(number_of_iteratons):
    # ----------------------------
    # Your code here
    p = np.random.normal(0.0, 1.0, M + 1)
    p = p / np.linalg.norm(p)
    selections = np.array(
        [
            MSE_loss(X, y, theta - eta * p),
            MSE_loss(X, y, theta),
            MSE_loss(X, y, theta + eta * p),
        ]
    )
    min_selection = np.min(selections)
    index = np.argmin(selections)

    if index == 0:
        theta -= eta * p
    elif index == 2:
        theta += eta * p

    # --------------------------
    loss = MSE_loss(X, y, theta)
    grad_free_losses.append(loss)
    eta *= 0.999
    if iter % 100 == 99:
        print(loss)

plt.plot(np.arange(len(second_order_losses)), second_order_losses, "black")
plt.plot(np.arange(len(gd_losses)), gd_losses, "red", label="gd")
plt.plot(np.arange(len(line_search_losses)), line_search_losses, "green")
plt.plot(
    np.arange(len(gd_with_momenutm_losses)),
    gd_with_momenutm_losses,
    "blue",
    label="gd_with_momentum",
)
plt.plot(
    np.arange(len(gd_with_nesterov_momenutm)),
    gd_with_nesterov_momenutm,
    "yellow",
    label="gd_with_momentum",
)
plt.plot(np.arange(len(coord_dec_losses)), coord_dec_losses, "purple")
plt.plot(np.arange(len(grad_free_losses)), grad_free_losses, "orange")

# Run this cell if you want to visualise a final model
yhat = X @ theta
plt.plot(x, y, "o", x, yhat, "red")

# plot the true function
plt.plot(np.linspace(-1, 1, 50), np.sin(2 * pi * 0.5 * np.linspace(-1, 1, 50)), "black")
