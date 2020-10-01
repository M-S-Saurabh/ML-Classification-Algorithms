from Utils import boston_50, boston_75, digits, make_80_20_splits
import numpy as np
import sys
import warnings
import matplotlib.pyplot as plt

def sigmoid(x): 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return 1/(1 + np.exp(-x))

def logisticLoss(y, preds):
    epsilon = 1e-5 
    loss = (-np.dot(y.T, np.log(preds + epsilon)) - np.dot((1 - y).T, np.log(1 - preds + epsilon))) / len(y)
    return loss

def classCounts(Y):
    Y_unique = np.unique(Y)
    for y in Y_unique:
        print("y:{} - count:{}".format(y, np.count_nonzero(Y == y)))

def visualizeLoss(loss):
    loss = np.array(loss)
    print("loss shape:",loss.shape)
    print(loss[0], np.mean(loss))
    plt.close('all')
    plt.figure()
    plt.plot(loss, 'b-')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.savefig('Loss.png')

def logisticReg(X_train, Y_train, X_test, Y_test, minibatch=25, learning_rate=0.05, iterations=2000):
    N, D = X_train.shape
    # Adding an extra feature with just ones
    X_train = np.hstack( (X_train, np.ones((N,1))) )
    X_test = np.hstack( (X_test, np.ones((X_test.shape[0],1))) )
    D += 1
    # Initialize w
    w = np.zeros(D,1)
    epoch = 0; start = 0; losses = []
    for it in range(iterations):
        if (start + minibatch) > N:
            np.random.shuffle(X_train)
            epoch += 1; start = 0
        # Choose a minibatch of points
        x_train = X_train[start : start+minibatch, :]
        y_train = Y_train[start : start+minibatch]
        y_train = y_train.reshape(len(y_train), 1)
        start += minibatch
        # Make predictions
        z = np.dot(x_train, w)
        preds = sigmoid(z)
        losses.append(logisticLoss(y_train, preds))
        # Calculate the gradients
        diff = preds - y_train
        dw = np.dot(x_train.T, diff) * (1 / minibatch)
        # Update w in opposite direction of gradient
        w -= dw * learning_rate
        # decay learning rate
        # if learning_rate > 0.001 and (it % 10 == 0):
        #     learning_rate *= 0.9
    # Visualize losses
    visualizeLoss(losses)
    # Test set predictions
    test_preds = sigmoid(np.dot(X_test, w)) > 0.5
    test_error_rate = np.count_nonzero(test_preds != Y_test) / len(Y_test)
    return test_error_rate
        

def normalize(X):
    # Assumes X is a NxD array
    return (X - np.mean(X, axis=0))

datasets = {
    'Boston50': boston_50,
    # 'Boston75': boston_75,
    # 'Digits': digits
}

params = {
    'Boston50': {'minibatch': 25, 'learning_rate': 0.02, 'iterations': 2000},
    'Boston75': {'minibatch': 30, 'learning_rate': 0.5, 'iterations': 10000}
}

def logisticRegression(num_splits=10, train_percent=[10, 25, 50, 75, 100]):
    for dataname, dataset in datasets.items():
        print()
        print("Dataset:", dataname)
        X, Y = dataset
        X = normalize(X)
        np.random.seed(42)
        error_rates = [[]] * num_splits
        for i in range(num_splits):
            print("Test-train split #", i)
            train_indices, test_indices = make_80_20_splits(Y)
            X_train, Y_train = X[train_indices], Y[train_indices]
            X_test, Y_test = X[test_indices], Y[test_indices]
            for pc in train_percent:
                cutoff = int(pc/100 * X_train.shape[0])
                error_pc = logisticReg(X_train[:cutoff,:], Y_train[:cutoff], X_test, Y_test, **params[dataname])
                error_rates[i].append(error_pc)
                
            print([" {}% data: {:.2f}% error".format(pc, err*100) for pc, err in zip(train_percent, error_rates[i])])

        pc_errors = np.mean(np.array(error_rates), axis=0)
        pc_std = np.var(np.array(error_rates), axis=0)
        print("Average error for each train_percent (across all splits)")
        print([" {}% data: {:.2f}% error".format(pc, err*100) for pc, err in zip(train_percent, pc_errors)])

if __name__ == "__main__":
    # xvec = [-3,-2,-1,0,1,2,3]
    # print("sigmoid of",xvec,"is:", sigmoid(xvec))
    logisticRegression(num_splits=1, train_percent=[100])
            