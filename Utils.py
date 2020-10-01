from sklearn.datasets import load_boston, load_digits
import numpy as np

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
        train_indices.append(indices[:(4*n//5)])
        test_indices.append(indices[(4*n//5):])
    train_indices = np.concatenate(train_indices)
    np.random.shuffle(train_indices)
    test_indices = np.concatenate(test_indices)
    np.random.shuffle(test_indices)
    return train_indices, test_indices

        
        
        