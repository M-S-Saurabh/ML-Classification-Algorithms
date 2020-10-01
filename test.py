from sklearn.datasets import load_iris
from Utils import make_80_20_splits
from logisticRegression import logisticReg
from sklearn.linear_model import LogisticRegression
import numpy as np

X, Y = load_iris(return_X_y=True)
X = X[(Y==0) | (Y == 1)]
Y = Y[(Y==0) | (Y == 1)]

train_indices, test_indices = make_80_20_splits(Y)
X_train, Y_train = X[train_indices], Y[train_indices]
X_test, Y_test = X[test_indices], Y[test_indices]

error_pc = logisticReg(X_train, Y_train, X_test, Y_test, minibatch=1, learning_rate=5e-2, iterations=2000)
print("Mine:", error_pc)

clf = LogisticRegression(random_state=0).fit(X_train, Y_train)
preds = clf.predict(X_test)
print("All:", np.all(preds == Y_test))
error_rate = np.count_nonzero(preds != Y_test) / len(Y_test)
print("SKLearn:",error_rate)