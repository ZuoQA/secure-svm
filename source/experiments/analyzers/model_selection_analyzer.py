from os import path, sep
import numpy as np
from numpy.core.records import array
import pandas as pd
import datetime
import json

import flp_svm
import flp_dual_svm_ls
import flp_dual_svm


def load_parameters(parameters, algorithm, X_train, y_train):
    params_list = list()
    for line in parameters.split("\n"):
        param_data = float(line.strip("\n").strip("[").strip("]"))
        params_list.append([param_data])

    params = np.array(params_list)

    if algorithm == "sgd":
        model = flp_svm.FlpSVM()
        model.load_params(params)
    elif algorithm == "smo":
        model = flp_dual_svm.FlpDualSVM()
        alphas = params[:params.shape[0] - 1]
        b = params[params.shape[0] - 1][0]
        model.load_parameters(alphas, b, X_train, y_train)
    elif algorithm == "ls":
        model = flp_dual_svm_ls.FlpDualLSSVM()
        alphas = params[:params.shape[0] - 1]
        b = params[params.shape[0] - 1][0]
        model.load_parameters(alphas, b, X_train, y_train)

    return model


def load_dataset(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :df.shape[1] - 1]
    y = df.iloc[:, df.shape[1] - 1]

    y = np.expand_dims(y, axis=1)

    return X.to_numpy(), y


def get_parameters(path_parameters):
    file_parameters = open(path_parameters, "r")
    parameters = file_parameters.read()

    file_parameters.close()
    return parameters

if __name__ == "__main__":
    algorithm = "sgd"
    path_parameters = "source/experiments/model_selection/svm_sgd_parameters.txt"
    path_train = "source/experiments/model_selection/datasets/toy_dataset_train.csv"
    path_test = "source/experiments/model_selection/datasets/toy_dataset_test.csv"
            
    X_train, y_train = load_dataset(path_train)
    X_test, y_test = load_dataset(path_test)

    parameters = get_parameters(path_parameters)

    model = load_parameters(parameters, algorithm, X_train, y_train)
    print("===> Secure SVM")
    sec_train_score = model.score(X_train, y_train)
    sec_test_score = model.score(X_test, y_test)
    print("Train acc =", sec_train_score)
    print("Test acc =", sec_test_score)

    print("===> Traditional SVM")
    time_a = datetime.datetime.now()
    model.fit(X_train, y_train)
    clean_time_date = datetime.datetime.now() - time_a
    clean_time = float(str(clean_time_date.seconds) + "." + str(clean_time_date.microseconds))
    print("Fit time =", clean_time)
    clean_train_score = model.score(X_train, y_train)
    print("Training accuracy =", clean_train_score)
    clean_test_score = model.score(X_test, y_test)
    print("Test accuracy =", clean_test_score)