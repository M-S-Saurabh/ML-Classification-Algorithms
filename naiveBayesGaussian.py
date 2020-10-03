import numpy as np
from collections import Counter

from Utils import boston_50, boston_75, digits
from Utils import problem4, gaussian_pdf

def NB_fit(X_train, Y_train, priors=None):
    Y_max = np.amax(Y_train)
    class_counts = Counter(Y_train)
    class_params = {}
    for y in range(Y_max+1):
        class_params[y] = {}
        # Class prior
        if priors is None:
            class_params[y]['prior'] = class_counts[y]/len(Y_train)
        else:
            class_params[y]['prior'] = priors[y]
        # Feature means and feature variances for each class
        x_train = X_train[(Y_train == y)]
        class_params[y]['means'] = np.mean(x_train, axis=0)
        epsilon = 1e-6
        class_params[y]['variances'] = np.var(x_train, axis=0) + epsilon
    return class_params

def NB_predict(class_params, X_test):
    N, D = X_test.shape
    class_posteriors = []
    for cl, params in class_params.items():
        likelihood = np.ones(N)
        for feature in range(D):
            x = X_test[:,feature].reshape(N,)
            mean = params['means'][feature]
            variance = params['variances'][feature]
            pdf = gaussian_pdf(x, mean, variance)
            likelihood *= pdf
        posterior = (likelihood * params['prior'])
        class_posteriors.append(posterior.reshape(N,1))
    class_posteriors = np.hstack(class_posteriors)
    preds = np.argmax(class_posteriors, axis=1)
    return preds

def naiveBayes(X_train, Y_train, X_test, Y_test, priors=None):
    class_params = NB_fit(X_train, Y_train, priors)
    preds = NB_predict(class_params, X_test)
    error_rate = np.count_nonzero(preds != Y_test) / len(Y_test)
    return error_rate

params = {
    'Boston50': {'priors':[0.5, 0.5]},
    'Boston75': {},
    'Digits': {},
}

def naiveBayesGaussian(num_splits=10, train_percent=[10, 25, 50, 75, 100]):
    return problem4(naiveBayes, num_splits, train_percent, params=params)

if __name__ == "__main__":
    # TODO cmdline arguments
    naiveBayesGaussian(num_splits=10, train_percent=[10, 25, 50, 75, 100])