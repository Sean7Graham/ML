import math
import numpy as np
import matplotlib.pyplot as plt

pi = math.pi

# generate 20 numbers from -1 to 1 with equal stepsize
x = np.linspace(-1, 1, 20)

# generate training target (noise contaminated!)
y = np.sin(2 * pi * 0.5 * x) + 0.3 * np.random.randn(x.size)

# Training data points
plt.plot(x, y, "ro")

# True function here
plt.plot(x, np.sin(pi * x))

X = [x]
X.append(np.ones(x.shape))
X = np.array(X).T
print(X.shape)
A = X.T.dot(X)
b = X.T.dot(y)

# Solve the regression on (X, y) and visualize the fitted function 
z = np.linalg.solve(A, b)
plt.plot(x, x * z[0] + z[1])

X = [x]
X.append(np.ones(x.shape))
X = np.array(X).T
print(X.shape)
A = X.T.dot(X)
b = X.T.dot(y)

# Solve the regression on (X, y) and visualize the fitted function 
z = np.linalg.solve(A, b)
plt.plot(x, x * z[0] + z[1])
