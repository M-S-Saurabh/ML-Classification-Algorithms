from logisticRegression import logisticRegression
from naiveBayesGaussian import naiveBayesGaussian
from Utils import datasets, plotCompareErrors

num_splits = 10
train_percent = [10, 25, 50, 75, 100]

LR_errors = logisticRegression(num_splits, train_percent)
NB_errors = naiveBayesGaussian(num_splits, train_percent)

names = ["LogisticReg", "NaiveBayes"]

for dataname in datasets.keys():
    error_list = []
    error_list.append(LR_errors[dataname])
    error_list.append(NB_errors[dataname])
    plotCompareErrors(names, dataname, error_list, train_percent)
