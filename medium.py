
import numpy as np
from Utils import boston_50, make_80_20_splits
from logisticRegression import visualizeLoss

def sigmoid(z):
    """Returns value of the sigmoid function."""
    return 1/(1 + np.exp(-z))

def cost_function(X, y, w):
    """
    Returns cost function and gradient

    Parameters
        X: m x (n+1) matrix of features
        y: m x 1 vector of labels
        w: (n+1) x 1 vector of weights
    Returns
        cost: value of cost function
        grad: (n+1) x 1 vector of weight gradients
    """
    
    m = len(y)
    h = sigmoid(np.dot(X, w))
    # print("h.shape", h.shape)
    epsilon = 1e-5
    cost = (1/m)*(-np.dot(y.T, np.log(h+epsilon)) - np.dot((1 - y).T, np.log(1 - h+epsilon)))
    # print("y.shape", y.shape)
    grad = (1/m)*np.dot(X.T, h - y)
    return cost, grad

def gradient_descent(X, y, w, alpha, num_iters):
    """
    Uses gradient descent to minimize cost function.
    
    Parameters
        X: m x (n+1) matrix of features
        y: m x 1 vector of labels
        w: (n+1) x 1 vector of weights
        alpha (float): learning rate
        num_iters (int): number of iterations
    Returns
        J: 1 x num_iters vector of costs
        w_new: (n+1) x 1 vector of optimized weights
        w_hist: (n+1) x num_iters matrix of weights
    """

    w_new = np.copy(w)
    w_hist = np.copy(w)
    m = len(y)
    J = np.zeros(num_iters)

    for i in range(num_iters):
        cost, grad = cost_function(X, y, w_new)
        # print(cost)
        w_new = w_new - alpha*grad
        w_hist = np.concatenate((w_hist, w_new), axis=1)
        J[i] = cost
    return J, w_new, w_hist

if __name__ == "__main__":
    X, Y = boston_50
    X = np.hstack( (X, np.ones((X.shape[0],1))) )

    train_indices, test_indices = make_80_20_splits(Y)
    X_train, Y_train = X[train_indices], Y[train_indices]
    X_test, Y_test = X[test_indices], Y[test_indices]

    Y_train = Y_train.reshape(len(Y_train),1)
    w = np.zeros((X.shape[1],1))
    J, w_train, w_hist = gradient_descent(X_train, Y_train, w, 5e-6, 2000)

    visualizeLoss(J)