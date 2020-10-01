
import sys
import numpy as np
from Utils import digits, cross_validation
import matplotlib.pyplot as plt
import itertools

def multivariate_gaussian_pdf(x, mu, cov):
    '''
    Taken from: https://stackoverflow.com/a/23101179
    
    Caculate the multivariate normal density (pdf)
    Keyword arguments:
        x = numpy array of a "d x 1" sample vector
        mu = numpy array of a "d x 1" mean vector
        cov = "numpy array of a d x d" covariance matrix
    '''
    D = len(mu)
    mu = mu.reshape(D, 1)
    x = x.reshape(D, 1)
    part1 = 1 / ( ((2* np.pi)**(D/2)) * (np.linalg.det(cov)**(1/2)) )
    part2 = (-0.5 * ((x-mu).T @ np.linalg.inv(cov) @ (x-mu)).real[0][0] )
    return float(part1 * np.exp(part2))

unique = itertools.count()
def visualize2D(X, Y, Y_unique):
    plt.figure()
    marker = itertools.cycle((',', '+', '.', 'o', '*', '<', '>', 'x', '1', '2')) 
    for y in Y_unique:
        plt.plot(X[(Y == y)], marker=next(marker), linestyle="None")
    plt.savefig("visualize/figure-{}.png".format(next(unique)))
    

def LDA2d(X_train, Y_train, X_test):
    D = X_train.shape[1]
    mu = X_train.mean(axis=0)
    Y_unique = np.unique(Y_train)
    for j, y_unq in enumerate(Y_unique):
        X_0 = X_train[(Y_train == y_unq)]
        mu_c = X_0.mean(axis=0)
        # Within class covariance Sw
        for i, x in enumerate(X_0):
            x_ = (x - mu_c).reshape(D,1)
            if i == j == 0:
                S_w = x_ @ x_.T
            else:
                S_w += (x_ @ x_.T)

        # Between class covariance Sb
        Nc = X_0.shape[0]
        mu_ = (mu_c - mu).reshape(D,1)
        if j == 0:
            S_b = Nc * (mu_ @ mu_.T)
        else:
            S_b += Nc * (mu_ @ mu_.T)

    # Transformation matrix #2 (By finding Eigen values of S_w^{-1} S_b)
    eigenValues, eigenVectors = np.linalg.eig(np.linalg.pinv(S_w) @ S_b)
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    A = eigenVectors[:,0:2] # Dx2

    X_train_new = X_train @ A # Nx2 matrix

    # Visualize training set in 2D space.
    visualize2D(X_train_new, Y_train, Y_unique)

    means, covariances = [], []
    # MLE of the Gaussians which fits each class
    for y_unq in Y_unique:
        X0_new = X_train_new[(Y_train == y_unq)]
        u = X0_new.mean(axis=0)
        cov = np.cov(X0_new.T)
        means.append(u); covariances.append(cov)

    train_preds = [ np.argmax([ multivariate_gaussian_pdf(x, u, cv) for u, cv in zip(means, covariances) ]) for x in X_train_new]

    X_test_new = X_test @ A # Nt x 2
    preds = [ np.argmax([ multivariate_gaussian_pdf(x, u, cv) for u, cv in zip(means, covariances) ]) for x in X_test_new]
    return train_preds, preds

if __name__ == "__main__":
    if(len(sys.argv) > 1): 
        try:
            num_crossval = int(sys.argv[1])
        except:
            print("Correct way to use arguments is: 'python LDA2dGaussGM.py 5'")
            sys.exit(1)
    print("Running 2-D LDA on Digits dataset...")
    train_errors, test_errors = cross_validation(LDA2d, digits[0], digits[1])
    print("Training Error mean:", np.mean(train_errors), " std:",np.var(train_errors) ** 0.5)
    print("Testing Error mean:", np.mean(test_errors), " std:",np.var(test_errors) ** 0.5)
