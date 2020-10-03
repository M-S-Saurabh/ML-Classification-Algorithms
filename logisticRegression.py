import numpy as np
import sys
import warnings

from Utils import boston_50, boston_75, digits
from Utils import problem4, visualizeLoss

def sigmoid(x): 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return 1/(1 + np.exp(-x))

def logisticLoss(y, preds):
    epsilon = 1e-5 # Added to remove errors on log(0)
    loss = (-np.dot(y.T, np.log(preds + epsilon)) - np.dot((1 - y).T, np.log(1 - preds + epsilon))) / len(y)
    return loss

def get_scores(w, X):
    X = np.hstack( (X, np.ones((X.shape[0],1))) )
    scores = sigmoid(np.dot(X, w))
    return scores

def multiclassLogisticReg(X_train, Y_train, X_test, Y_test, learning_rate=0.05, iterations=2000):
    Y_max = np.amax(Y_train)
    scores = []
    for y_unq in range(Y_max+1):
        y_train = (Y_train == y_unq).astype('int')
        w_ = binaryLogisticReg(X_train, y_train, learning_rate=learning_rate, iterations=iterations, returnModel=True)
        if len(scores) == 0:
            scores = get_scores(w_, X_test)
        else:
            scores = np.hstack((scores, get_scores(w_, X_test)))
    preds = np.argmax(scores, axis=1)
    error_rate = np.count_nonzero(preds != Y_test) / len(Y_test)
    return error_rate

def binaryLogisticReg(X_train, Y_train, X_test=None, Y_test=None, learning_rate=0.05, iterations=2000, returnModel=False):
    N, D = X_train.shape
    # Adding an extra feature with just ones
    X_train = np.hstack( (X_train, np.ones((N,1))) )
    D += 1
    Y_train = Y_train.reshape(len(Y_train), 1)
    # Initialize w
    w = np.zeros((D,1))
    epoch = 0; start = 0; losses = np.zeros(iterations)
    for it in range(iterations):
        # Make predictions
        z = np.dot(X_train, w)
        preds = sigmoid(z)
        losses[it] = logisticLoss(Y_train, preds)
        # Calculate the gradients
        diff = preds - Y_train
        dw = np.dot(X_train.T, diff) * (1 / len(Y_train))
        # Update w in opposite direction of gradient
        w = w - dw * learning_rate
        # decay learning rate
        # if learning_rate > 0.001 and (it % 10 == 0):
        #     learning_rate *= 0.9
    # Visualize losses
    # visualizeLoss(losses)

    if (X_test is not None) and (Y_test is not None):
        # Test set predictions
        X_test = np.hstack( (X_test, np.ones((X_test.shape[0],1))) )
        Y_test = Y_test.reshape(len(Y_test), 1)
        test_preds = sigmoid(np.dot(X_test, w)) > 0.5
        test_error_rate = np.count_nonzero(test_preds != Y_test) / len(Y_test)
        return test_error_rate

    if returnModel:
        return w.copy()

def logisticRegression1(X_train, Y_train, X_test, Y_test, learning_rate=0.05, iterations=2000):
    if(np.amax(Y_train) > 1):
        return multiclassLogisticReg(X_train, Y_train, X_test, Y_test, learning_rate=learning_rate, iterations=iterations)
    else:
        return binaryLogisticReg(X_train, Y_train, X_test, Y_test, learning_rate=learning_rate, iterations=iterations)


params = {
    'Boston50': {'learning_rate': 6.85e-5, 'iterations': 2000}, # Smooth loss: 1e-6
    'Boston75': {'learning_rate': 4.85e-6, 'iterations': 10000}, # 4.85e-6
    'Digits': {'learning_rate': 5e-6, 'iterations': 2000},
}

def logisticRegression(num_splits=10, train_percent=[10, 25, 50, 75, 100]):
    return problem4(logisticRegression1, num_splits, train_percent, params)

if __name__ == "__main__":
    # TODO cmdline arguments
    logisticRegression(num_splits=10, train_percent=[10, 25, 50, 75, 100])
            