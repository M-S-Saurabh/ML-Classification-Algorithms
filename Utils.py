from sklearn.datasets import load_boston, load_digits
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def make_boston_datasets():
    boston_data_bunch = load_boston()
    data, target = boston_data_bunch.data, boston_data_bunch.target

    target_values = np.sort(target.copy())
    t_50 = target_values[len(target_values)//2]
    t_75 = target_values[3*len(target_values)//4]

    y_50 = (target_values >= t_50).astype('int32')
    y_75 = (target_values >= t_75).astype('int32')

    return (data, y_50), (data, y_75)

boston_50, boston_75 = make_boston_datasets()

digits_data_bunch = load_digits()
digits = (digits_data_bunch.data, digits_data_bunch.target)

def gaussian_pdf(x, mean, var):
    num = np.exp(-0.5 * (x-mean)**2 / var)
    den = (2 * np.pi * var) ** 0.5
    return (num / den)

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

def cross_validation(method_to_call, train_X, train_Y, num_crossval=10):
    # Create 10 parts of data
    X_splits, Y_splits = [], []
    y_values = np.unique(train_Y)
    for y_value in y_values:
        y_indices = (train_Y == y_value).nonzero()[0]
        np.random.shuffle(y_indices)
        y_splits = np.array_split(y_indices, num_crossval)
        if len(X_splits) == 0 and len(Y_splits) == 0:
            X_splits = [train_X[split] for split in y_splits]
            Y_splits = [train_Y[split] for split in y_splits]
        else:
            for i, split in enumerate(y_splits):
                X_splits[i] = np.vstack((X_splits[i], train_X[split]))
                Y_splits[i] = np.concatenate((Y_splits[i], train_Y[split]))
    
    train_errors, test_errors = [], []
    # Loop over parts selecting one as test set each time.
    for i in range(len(X_splits)):
        X_test = X_splits[i]
        Y_test = Y_splits[i]

        X_train = np.vstack([X for j, X in enumerate(X_splits) if j != i])
        Y_train = np.concatenate([Y for j, Y in enumerate(Y_splits) if j != i])

        # TODO: call function and train, perform tests, report results
        train_preds, preds = method_to_call(X_train, Y_train, X_test)
        train_error_count = np.count_nonzero(Y_train != train_preds)
        train_errors.append(train_error_count/len(Y_train))

        error_count = np.count_nonzero(Y_test != preds)
        test_errors.append(error_count/len(Y_test))
    return train_errors, test_errors

def make_80_20_splits(Y):
    Y_unique = np.unique(Y)
    train_indices, test_indices = [], []
    for y_unq in Y_unique:
        indices = np.argwhere(Y == y_unq)
        n = len(indices)
        indices = indices.reshape(n,)
        np.random.shuffle(indices)
        train_indices.append(indices[:(4*n//5)])
        test_indices.append(indices[(4*n//5):])
    train_indices = np.concatenate(train_indices)
    np.random.shuffle(train_indices)
    test_indices = np.concatenate(test_indices)
    np.random.shuffle(test_indices)
    return train_indices, test_indices
        
def classCounts(Y):
    Y_unique = np.unique(Y)
    for y in Y_unique:
        print("y:{} - count:{}".format(y, np.count_nonzero(Y == y)))

def plotErrors(methodname, dataname, errors, train_percent):
    means = np.mean(errors, axis=0)
    std = np.std(errors, axis=0)
    plt.figure()
    plt.errorbar(train_percent, means, yerr=std, capsize=5, fmt='o-', ecolor='r')
    plt.title('{} error rate: {} data'.format(methodname, dataname))
    plt.xlabel('Training data percentage')
    plt.ylabel('Test set error rate')
    filename = "{}_{}_error.png".format(methodname, dataname)
    plt.savefig("visualize/{}".format(filename))

def plotCompareErrors(method_names, dataname, error_matrices, train_percent):
    plt.figure()
    fig,(ax1)=plt.subplots(1,1)
    for i, (methodname, errors) in enumerate(zip(method_names, error_matrices)):
        means = np.mean(errors, axis=0)
        std = np.std(errors, axis=0)
        if i == 0:
            ax1.errorbar(train_percent, means, yerr=std, label=methodname, fmt='o-', color='blue', capsize=5, ecolor='blue')
        else:
            ax1.errorbar(train_percent, means, yerr=std, label=methodname, fmt='o--', color='red', capsize=5, ecolor='red')
    ax1.legend(loc='upper right')
    ax1.set_title('{} v {} error rate: {} data'.format(*method_names, dataname))
    ax1.set_xlabel('Training data percentage')
    ax1.set_ylabel('Test set error rate')
    filename = "{}_vs_{}_error - {}.png".format(*method_names, dataname)
    fig.savefig("visualize/{}".format(filename))

def visualizeLoss(loss):
    loss = np.array(loss)
    plt.close('all')
    plt.figure()
    plt.plot(loss, 'b-')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Logistic regression: Loss across iterations')
    plt.savefig('visualize/Loss.png')

def normalize_zeromean(X):
    # Assumes X is a NxD array
    return (X - np.mean(X, axis=0))

def normalize_01(X):
    # Assumes X is a NxD array
    X_norm = (X - np.amin(X, axis=0)) / (np.amax(X, axis=0) - np.amin(X, axis=0))
    return X_norm


datasets = {
    'Boston50': boston_50,
    'Boston75': boston_75,
    'Digits': digits
}

def problem4(method, num_splits=10, train_percent=[10, 25, 50, 75, 100], params=None):
    dataset_errors = {}
    for dataname, dataset in datasets.items():
        print()
        print("---------------------------------------------------")
        print("---------Dataset: {}---------".format(dataname))
        X, Y = dataset
        X = normalize_zeromean(X)
        np.random.seed(42)
        error_rates = np.zeros((num_splits, len(train_percent)))
        prev_train_indices = []
        for i in range(num_splits):
            print("---------------------")
            print("Test-train split #", i+1)
            train_indices, test_indices = make_80_20_splits(Y)
            X_train, Y_train = X[train_indices], Y[train_indices]
            X_test, Y_test = X[test_indices], Y[test_indices]
            for j, pc in enumerate(train_percent):
                cutoff = int(pc/100 * X_train.shape[0])
                error_pc = method(X_train[:cutoff,:], Y_train[:cutoff], X_test, Y_test, **params[dataname])
                error_rates[i][j] = error_pc
            print(" ".join(["{}% data: {:.2f}%;".format(pc, err*100) for pc, err in zip(train_percent, error_rates[i])]))

        pc_errors = np.mean(np.array(error_rates), axis=0)
        print("--------------------------------------------------------")
        print("Average error for each train_percent (across all splits):")
        print(" ".join(["{}% data: {:.2f}%;".format(pc, err*100) for pc, err in zip(train_percent, pc_errors)]))
        # Plot avg error rates for each train_percent
        plotErrors(method.__name__,dataname, error_rates, train_percent)
        dataset_errors[dataname] = error_rates
    return dataset_errors
        