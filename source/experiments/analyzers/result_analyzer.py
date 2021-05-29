import numpy as np
from numpy.core.records import array
import pandas as pd
import datetime

import flp_svm
import flp_dual_svm_ls
import flp_dual_svm


def load_parameters(path, algorithm, X_train, y_train):
    params_file = open(path, "r")
    params_list = list()
    for line in params_file.readlines():
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


if __name__ == "__main__":
    experiment_list = [
        "test-40r-2c",
        "test-40r-2c",
        "test-40r-2c",
        "test-40r-2c",
        "test-40r-2c",

        "test-50r-2c",
        "test-60r-2c",
        "test-70r-2c",
        "test-80r-2c",
    ]

    experiment = "real_experiment"
    dataset_name = "toy_dataset"
    
    path_train = "source/experiments/" + experiment + "/datasets/" + dataset_name + "_train.csv"
    path_test = "source/experiments/" + experiment + "/datasets/" + dataset_name + "_test.csv"
    
    path_parameters = "source/experiments/" + experiment + "/svm_ls_parameters.txt"
    algorithm = "ls"
    
    X_train, y_train = load_dataset(path_train)
    X_test, y_test = load_dataset(path_test)

    model = load_parameters(path_parameters, algorithm, X_train, y_train)
    print("===> Secure SVM")
    print("Train acc =", model.score(X_train, y_train))
    print("Test acc =", model.score(X_test, y_test))

    print("===> Traditional SVM")
    time_a = datetime.datetime.now()
    model.fit(X_train, y_train)
    print("Fit time =", datetime.datetime.now() - time_a)
    training_score = model.score(X_train, y_train)
    print("Training accuracy =", training_score)
    test_score = model.score(X_test, y_test)
    print("Test accuracy =", training_score)
