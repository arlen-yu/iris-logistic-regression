from scipy.optimize import minimize
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""Data from: https://archive.ics.uci.edu/ml/datasets/Iris"""

SEPAL_LENGTH = 'sepal_length'
SEPAL_WIDTH = 'sepal_width'
PETAL_LENGTH = 'petal_length'
PETAL_WIDTH = 'petal_width'
SPECIES = 'species'
SETOSA = 'setosa'


def describe_species(data):
    """ Provides intuition on what features to use when comparing species"""
    print(data[setosa_mask])
    print(data[versicolor_mask])
    print(data[virginica_mask])

    plt.scatter(data[setosa_mask][SEPAL_LENGTH],
                data[setosa_mask][PETAL_WIDTH], marker='o', c='blue')
    plt.scatter(data[versicolor_mask][SEPAL_LENGTH],
                data[versicolor_mask][PETAL_WIDTH], marker='^', c='red')
    plt.scatter(data[virginica_mask][SEPAL_LENGTH],
                data[virginica_mask][PETAL_WIDTH], marker='s', c='green')
    plt.xlabel('Sepal Length')
    plt.ylabel('Petal Width')
    plt.show()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


"""FOR GRADIENT DESCENT"""


def compute_cost(X, y, theta):
    J = 0
    _sigmoid = np.vectorize(sigmoid)
    h_vals = _sigmoid(X.dot(theta))

    for i in range(1, len(X)):
        J += (h_vals[i] - y[i])**2

    J = J/(2*len(X))

    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    J_history = np.zeros(num_iters)

    # TODO vectorize
    for iter in np.arange(num_iters):
        tmp_theta = theta.copy()
        inside_sum = sigmoid(theta.T.dot(X.T)) - y.T

        for i in range(0, 3):
            tmp_theta[i] = theta[i] - alpha * \
                (1/m) * sum(np.dot(inside_sum, X[:, i]))

        theta = tmp_theta.copy()
        J_history[iter] = compute_cost(X, y, theta)

    return (theta, J_history)


"""FOR ADVANCED OPTIMIZATION"""


def cost_function(theta, X, y):
    m = len(y)
    _sigmoid = np.vectorize(sigmoid)
    h = _sigmoid(X.dot(theta))
    J = (1/m) * (-1 * y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h)))

    # if np.isnan(J[0]):
    #     return(np.inf)
    return J[0]


def gradient_function(theta, X, y):
    m = len(y)
    _sigmoid = np.vectorize(sigmoid)
    h = _sigmoid(np.dot(np.transpose(theta), np.transpose(X)))
    grad = (1/m) * np.dot(np.subtract(h, np.transpose(y)), X)
    return grad.flatten()


def predict(theta, X):
    m = len(X)

    _sigmoid = np.vectorize(sigmoid)
    h = _sigmoid(np.dot(X, theta))
    p = []

    for i in range(1, m):
        if h[i] >= 0.5:
            p.append(1)
        else:
            p.append(0)

    return np.array(p)


if __name__ == "__main__":
    data = pd.read_csv('data.csv')
    setosa_mask = data[SPECIES] == 'setosa'
    versicolor_mask = data[SPECIES] == 'versicolor'
    virginica_mask = data[SPECIES] == 'virginica'

    # describe_species(data)

    # Set up data
    m = len(data)
    X = np.c_[np.ones(m), data.iloc[:, 0], data.iloc[:, 3]]
    y = np.c_[[1 if name == SETOSA else 0 for name in data[SPECIES]]]

    theta = np.c_[np.zeros(3)]  # linear decision boundary

    if len(sys.argv) <= 1 or sys.argv[1] != 'advanced':

        # Run gradient descent
        print('Running gradient descent')

        # Choose alpha value
        alpha = 0.1
        num_iters = 700

        # Init theta and run gradient descent
        theta, J_history = gradient_descent(X, y, theta, alpha, num_iters)

        # Plot the convergence graph
        plt.plot(np.arange(len(J_history)), J_history)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.show()

        # Display gradient descent's result
        print('\nTheta computed from gradient descent:')
        print(theta)

        # Plot the decision boundary
        plt.scatter(data[setosa_mask][SEPAL_LENGTH],
                    data[setosa_mask][PETAL_WIDTH], marker='o', c='blue')
        plt.scatter(data[versicolor_mask][SEPAL_LENGTH],
                    data[versicolor_mask][PETAL_WIDTH], marker='^', c='red')
        plt.scatter(data[virginica_mask][SEPAL_LENGTH],
                    data[virginica_mask][PETAL_WIDTH], marker='s', c='green')

        plot_x = [min(X[:, 1]) - 2, max(X[:, 1]) + 2]
        plot_y = (-1/theta[2]) * (theta[1]*plot_x + theta[0])

        plt.plot(plot_x, plot_y)

        plt.show()

        print('\nAccuracy of hypothesis compared to training data:')
        p = predict(theta, X)
        print(np.mean([1 if p[i] == y[i] else 0 for i in range(0, m-1)]))
    else:

        J = cost_function(theta, X, y)
        grad = gradient_function(theta, X, y)

        initial_theta = np.c_[np.zeros(3)]

        print(cost_function(initial_theta, X, y))
        res = minimize(cost_function, initial_theta, args=(
            X, y), method=None, jac=gradient_function, options={'maxiter': 400})

        print(res)

        theta = res.x.T

        # Plot the decision boundary
        plt.scatter(data[setosa_mask][SEPAL_LENGTH],
                    data[setosa_mask][PETAL_WIDTH], marker='o', c='blue')
        plt.scatter(data[versicolor_mask][SEPAL_LENGTH],
                    data[versicolor_mask][PETAL_WIDTH], marker='^', c='red')
        plt.scatter(data[virginica_mask][SEPAL_LENGTH],
                    data[virginica_mask][PETAL_WIDTH], marker='s', c='green')

        theta = np.c_[theta]
        plot_x = np.array([min(X[:, 1]) - 2, max(X[:, 1]) + 2])
        plot_y = (-1/theta[2]) * (theta[1]*plot_x + theta[0])

        plt.plot(plot_x, plot_y)

        plt.show()

        print('Theta computed using advanced optimization:')
        print(theta)

        print('\nAccuracy of hypothesis compared to training data:')
        p = predict(theta, X)
        print(np.mean([1 if p[i] == y[i] else 0 for i in range(0, m-1)]))

    # Predict the probability of a flower with sepal length of 5.3
    # and a petal width of 1 of being a setosa
    prob = sigmoid(np.array([1, 5.3, 0.8]).dot(theta))
    print('\nFor a flower with sepal length of 5.3 and a petal length of 1 we predict a ' +
          str(int(prob * 100)) + '% chance of it being a setosa')
