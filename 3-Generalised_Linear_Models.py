# The original dataset is large, below is a cleaned subset.
# loaded the cleaned subset of bikeshare data, it is already a numpy matrix
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from google.colab import drive


# grab data from google drive
drive.mount("/content/drive")

X = np.loadtxt(
    "/content/drive/MyDrive/Colab Notebooks/ML CWM/subsetbikeshare.txt", delimiter=","
)

print(X.shape)
np.random.shuffle(X)
# I already put the target variable the last column
Y = X[:, -1]
X = X[:, :-1]
X = np.hstack((X, np.ones((X.shape[0], 1))))

valn = int(0.1 * X.shape[0])
Yval = Y[:valn]
Xval = X[:valn]

X = X[valn:]
Y = Y[valn:]

exit
# print(Y[:10])
# TODO: complete the code below to compute gradient
def grad(w):
    return (np.sum(X.T @ (Y - np.exp(X @ w)))) / len(X)


w = np.zeros(X.shape[1])

for i in range(3000):
    w = w + 0.0001 * grad(w)
    if i % 100 == 0:
        print(" validation err is :: ", np.sqrt(np.mean(np.square(Xval.dot(w) - Yval))))
        print(" train err is :: ", np.sqrt(np.mean(np.square(X.dot(w) - Y))))

# The original dataset is large, below I provide a cleaned subset.
# load the cleaned subset of bikeshare data, it is already a numpy matrix
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from google.colab import drive

# you can get data from: https://drive.google.com/file/d/1OVUZ9hEag6OjPD3gv8zfEtY-rD_pv7yg/view?usp=sharing
# and then copy it to your own google drive

# grab data from google drive
drive.mount("/content/drive")
# the first argument is the location of your file, in my google drive,
# I put the dataset under the directory cwm-python-aiml
X = np.loadtxt(
    "/content/drive/MyDrive/Colab Notebooks/ML CWM/subsetbikeshare.txt", delimiter=","
)
print(X.shape)
# I already put the target variable the last column
Y = X[:, -1]
X = X[:, :-1]

# init an empty model
poisson_model = tf.keras.models.Sequential()
# add input layer
poisson_model.add(tf.keras.Input(shape=(114)))
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
poisson_model.add(
    tf.keras.layers.Dense(
        units=1, kernel_regularizer=tf.keras.regularizers.L2(1e-2), activation=tf.exp
    )
)
# for adding regularizers, see https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/Regularizer

# take a look the summary of the model
poisson_model.summary()

# reduce mean function returns the mean of the input vector
def loss_fn(y_true, y_pred):
    return -tf.reduce_mean(y_true * tf.squeeze(tf.math.log(y_pred)) - y_pred)


modelmetric = tf.keras.metrics.RootMeanSquaredError()

# regarding compile function, https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile
poisson_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=loss_fn,
    metrics=[modelmetric],
)

# fit the poisson model, https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
# batch_size is the number of samples used to compute gradient
history = poisson_model.fit(
    X,
    Y,
    epochs=100,
    batch_size=64,
    verbose=0,
    # Calculate validation results on 20% of the training data.
    # you can also feed your own validation set by setting validation_data=...
    validation_split=0.2,
)

print(history.history)
print([key for key in history.history])

# visualize the training progress
def plot_loss(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    # plt.ylim([0, 10000])
    plt.xlabel("Epoch")
    plt.ylabel("loss [bike rental]")
    plt.legend()
    plt.grid(True)


plt.figure()
plot_loss(history)


def plot_err(history):
    plt.plot(history.history[modelmetric.name], label=modelmetric.name)
    plt.plot(
        history.history["val_" + modelmetric.name], label="val_" + modelmetric.name
    )
    # plt.ylim([0, 10000])
    plt.xlabel("Epoch")
    plt.ylabel(modelmetric.name + " [bike rental]")
    plt.legend()
    plt.grid(True)


plt.figure()
plot_err(history)

# https://www.tensorflow.org/tutorials/keras/regression has more examples
# for building regression models

print(X.shape)
# init an empty model
lr_model = tf.keras.models.Sequential()
# add input layer
lr_model.add(tf.keras.Input(shape=(114)))
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense

# TODO: modify the activation here
lr_model.add(
    tf.keras.layers.Dense(
        units=1, kernel_regularizer=tf.keras.regularizers.L2(1e-2), activation=None
    )
)

# TODO:
def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - tf.squeeze(y_pred)))


modelmetric = tf.keras.metrics.RootMeanSquaredError()

lr_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=loss_fn,
    metrics=[modelmetric],
)

history = lr_model.fit(X, Y, epochs=100, batch_size=64, verbose=0, validation_split=0.2)

print(history.history)
print([key for key in history.history])

# visualize the training progress
def plot_loss(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    # plt.ylim([0, 10000])
    plt.xlabel("Epoch")
    plt.ylabel("loss [bike rental]")
    plt.legend()
    plt.grid(True)


plt.figure()
plot_loss(history)


def plot_err(history):
    plt.plot(history.history[modelmetric.name], label=modelmetric.name)
    plt.plot(
        history.history["val_" + modelmetric.name], label="val_" + modelmetric.name
    )
    # plt.ylim([0, 10000])
    plt.xlabel("Epoch")
    plt.ylabel(modelmetric.name + " [bike rental]")
    plt.legend()
    plt.grid(True)


plt.figure()
plot_err(history)

# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0


model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        ## TODO: add one line here for output, you need to check out below loss function
        tf.keras.layers.Dense(units=10, activation=None),
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    ),  # check what from_logits means
    metrics=["accuracy"],
)

# complete the rest of code
# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=32)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def log_likelihood_gradient_hessian(X, y, w):
    n = X.shape[0]
    gradient = np.zeros_like(w)
    hessian = np.zeros((w.shape[0], w.shape[0]))

    for i in range(n):
        x = X[i]
        y_i = y[i]
        z = np.dot(w, x)
        a = 1 + z**2
        p = 0.5 * (1 + z / np.sqrt(a))

        # Compute gradient
        gradient += y_i * x / (a**0.5 * (1 + np.sqrt(a)))
        gradient -= (1 - y_i) * x / (a**0.5 * (1 + np.sqrt(a)))

        # Compute Hessian matrix
        hessian += (y_i / (a**0.5 * (1 + np.sqrt(a)))) * np.outer(x, x)
        hessian -= (1 - y_i) / (a**0.5 * (1 + np.sqrt(a))) * np.outer(x, x)

    return gradient, hessian


# Example usage
X = np.array([[1, 2], [3, 4], [5, 6]])  # Input features
y = np.array([1, 0, 1])  # Labels
w = np.array([0.5, -0.5])  # Parameter vector

gradient, hessian = log_likelihood_gradient_hessian(X, y, w)
print("Gradient:")
print(gradient)
print("\nHessian matrix:")
print(hessian)
