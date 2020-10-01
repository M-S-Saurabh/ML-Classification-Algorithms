
import sys
import numpy as np
from Utils import boston_50, cross_validation
import matplotlib.pyplot as plt

def gaussian_pdf(mean, var, x):
    num = np.exp(-0.5 * (x-mean)**2 / var)
    den = (2 * np.pi * var) ** 0.5
    return (num / den)

def LDA1d(X_train, Y_train, X_test):
    D = X_train.shape[1]
    X_0 = X_train[(Y_train == 0)]
    m1 = X_0.mean(axis=0).reshape(D,1)
    for i, x in enumerate(X_0):
        xT = x.reshape(D,1)
        if i == 0:
            S_w = (xT - m1) @ (xT - m1).T
        else:
            S_w += (xT - m1) @ (xT - m1).T
    
    X_1 = X_train[(Y_train == 1)]
    m2 = X_1.mean(axis=0).reshape(D,1)
    for i, x in enumerate(X_1):
        xT = x.reshape(D,1)
        S_w += (xT - m2) @ (xT - m2).T

    S_b = (m2 - m1) @ (m2 - m1).T

    # Transformation matrix
    A = np.linalg.inv(S_w) @ (m2 - m1)

    X0_new = X_0 @ A; X1_new = X_1 @ A
    # MLE of the Gaussians which fits each class
    u0 = X0_new.mean(); v0 = np.var(X0_new)
    u1 = X1_new.mean(); v1 = np.var(X1_new)

    X_train_new = (X_train @ A).reshape(len(X_train),)
    train_preds = np.array([(gaussian_pdf(u0, v0, x) < gaussian_pdf(u1, v1, x)) for x in X_train_new])

    X_test_new = (X_test @ A).reshape(len(X_test),)
    Y_test = np.array([(gaussian_pdf(u0, v0, x) < gaussian_pdf(u1, v1, x)) for x in X_test_new])
    return train_preds, Y_test

if __name__ == "__main__":
    num_crossval = 10
    if(len(sys.argv) > 1): 
        try:
            num_crossval = int(sys.argv[1])
        except:
            print("Correct way to use arguments is: 'python LDA1dThres.py 5'")
            sys.exit(1)
    print("Running 1-D LDA on Boston-50 dataset...")
    train_errors, test_errors = cross_validation(LDA1d, boston_50[0], boston_50[1], num_crossval=num_crossval)
    print("Training Error mean:", np.mean(train_errors), " std:", np.var(train_errors) ** 0.5)
    print("Testing Error mean:", np.mean(test_errors), " std:", np.var(test_errors) ** 0.5)
