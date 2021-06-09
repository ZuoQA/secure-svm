from os import path, sep
import numpy as np
from numpy.core.records import array
import pandas as pd
import datetime
import json

import flp_svm
import flp_dual_svm_ls
import flp_dual_svm


with open("source/experiments/config.json") as config_file:
    config = json.load(config_file)
with open("source/experiments/experiment_info.json") as file_info:
    data_experiments = json.load(file_info)


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
    file_content = file_parameters.read()
    file_parts = file_content.split("--\n")

    parameters = file_parts[2]
    parameters = parameters.strip("\n").strip(" ")

    file_parameters.close()
    return parameters
    


def get_execution_info(path_parameters):
    file_parameters = open(path_parameters, "r")
    file_content = file_parameters.read()
    file_parts = file_content.split("--\n")
    execution_info_part = file_parts[3]

    execution_info_splitted = execution_info_part.split("\n")
    time = float(execution_info_splitted[0].lstrip("Time = ").rstrip(" seconds"))
    data = float(execution_info_splitted[1].lstrip("Data sent = ").rstrip(" MB"))
    global_data = float(execution_info_splitted[2].lstrip("Global data sent = ").rstrip(" MB"))
    
    file_parameters.close()
    return time, data, global_data


if __name__ == "__main__":
    experiment_list = [
        "test-40r-2c",
        "test-50r-2c",
        "test-60r-2c",
        "test-70r-2c",
        "test-80r-2c",
        "test-90r-2c",

        "test-100r-2c",
        "test-100r-3c",
        "test-100r-4c",
        "test-100r-5c",
        "test-100r-6c",
        "test-100r-7c",
        "test-100r-8c",
        "test-100r-9c",
        "test-100r-10c",

        "test_class_sep_040",
        "test_class_sep_050",
        "test_class_sep_060",
        "test_class_sep_070",
        "test_class_sep_080",
        "test_class_sep_090",
        "test_class_sep_100",
        "test_class_sep_110"
    ]

    algorithm = "ls"

    result_data = []
    for experiment in experiment_list:
        first_dataset = False
        

        for n_execution in range(data_experiments[experiment]["n_executions"]):
            if not first_dataset or data_experiments[experiment]["change_dataset"]:
                dataset_name = "toy_dataset"
                path_train = config["experiments_path"] + experiment + "/datasets/" + dataset_name + "_train_" + str(n_execution) + ".csv"
                path_test = config["experiments_path"] + experiment + "/datasets/" + dataset_name + "_test_" + str(n_execution) + ".csv"
                first_dataset = True
            
            path_parameters = config["experiments_path"] + experiment + "/ouput_secure_" + str(n_execution) + ".txt"
            
            X_train, y_train = load_dataset(path_train)
            X_test, y_test = load_dataset(path_test)

            parameters = get_parameters(path_parameters)
            sec_time, data_sent, global_data_sent = get_execution_info(path_parameters)

            model = load_parameters(parameters, algorithm, X_train, y_train)
            print("#############", experiment, "-", n_execution, "#############")
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


            experiment_results = [
                data_experiments[experiment]["n_rows"],
                data_experiments[experiment]["n_columns"],
                data_experiments[experiment]["class_sep"],
                n_execution,
                sec_train_score,
                sec_test_score,
                clean_train_score,
                clean_test_score,
                sec_time,
                clean_time,
                data_sent,
                global_data_sent
            ]
                    
            result_data.append(experiment_results)
    

    headers = [
        "Rows",
        "Columns",
        "Class sep",
        "N execution",
        "Train acc sec",
        "Test acc sec",
        "Train acc clean",
        "Test acc clean",
        "Time sec",
        "Time clean",
        "Data sent",
        "Global data sent",
    ]

    df_results = pd.DataFrame(
        data=np.array(result_data),
        columns=headers
    )

    df_results.to_csv(config["experiments_path"] + "results_repeated.csv")
    
    """ # Real experiment
    experiment = "real_experiment"
    dataset_name = "real_dataset"
        
    path_train = "source/experiments/" + experiment + "/datasets/" + dataset_name + "_train.csv"
    path_test = "source/experiments/" + experiment + "/datasets/" + dataset_name + "_test.csv"
    
    path_parameters = "source/experiments/" + experiment + "/svm_ls_parameters.txt"
    algorithm = "ls"
    
    X_train, y_train = load_dataset(path_train)
    X_test, y_test = load_dataset(path_test)

    model = load_parameters(path_parameters, algorithm, X_train, y_train)
    print("#############", experiment, "#############")
    print("===> Secure SVM")
    print("Train acc =", model.score(X_train, y_train))
    print("Test acc =", model.score(X_test, y_test))

    print("===> Traditional SVM")
    model = flp_dual_svm_ls.FlpDualLSSVM(
        lambd=8,
        lr=1e-2,
        max_iter=80,
        kernel="linear",
        tolerance=1e-7,
    )
    time_a = datetime.datetime.now()
    model.fit(X_train, y_train)
    print("Fit time =", datetime.datetime.now() - time_a)
    training_score = model.score(X_train, y_train)
    print("Training accuracy =", training_score)
    test_score = model.score(X_test, y_test)
    print("Test accuracy =", test_score) """