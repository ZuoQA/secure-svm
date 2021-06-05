import subprocess
import json
import math
import numpy
from scipy.sparse import data
from sklearn.utils import _list_indexing

import dataset_generator


# Load information
with open("source/experiments/config.json") as config_file:
    config = json.load(config_file)
with open("source/experiments/experiment_info.json") as file_info:
    data_experiments = json.load(file_info)


def execute_experiment(experiment): 
    compile_library(experiment)
    compile_bytecode(experiment)

    first_dataset = False
    for n_execution in range(data_experiments[experiment]["n_executions"]):
        if not first_dataset or data_experiments[experiment]["change_dataset"]:
            generate_dataset_experiment(experiment, n_execution)
            first_dataset = True

        cd_command = "cd " + config["mp_spdz_path"]
        command = cd_command + " && " + config["sec_execution_command"]
        print("Running:", command)
        result = subprocess.run([command], stdout=subprocess.PIPE, shell=True)
        result.check_returncode()

        result_str = result.stdout.decode('utf-8')
        save_experiment_output(experiment, result_str, n_execution)


def compile_library(experiment):
    copy_command = "cp -rf " + config["experiments_path"] + experiment + "/CONFIG.mine " + config["mp_spdz_path"] + "CONFIG.mine"
    print("Running:", copy_command)
    result = subprocess.run([copy_command], stdout=subprocess.PIPE, shell=True)
    result.check_returncode()

    compile_command = "cd " + config["mp_spdz_path"] + " && " + config["compile_command"]
    print("Running:", compile_command)
    result = subprocess.run([compile_command], stdout=subprocess.PIPE, shell=True)
    result.check_returncode()


def compile_bytecode(experiment):
    # cd MP-SPDZ ./compile.py -R 144 secure_dual_ls_svm
    copy_command = "cp -rf " + config["experiments_path"] + experiment + "/secure_dual_ls_svm.mpc " + config["mp_spdz_path"] + "/Programs/Source/secure_dual_ls_svm.mpc"
    print("Running:", copy_command)
    result = subprocess.run([copy_command], stdout=subprocess.PIPE, shell=True)
    result.check_returncode()

    compile_command = "cd " + config["mp_spdz_path"] + " && " + "./compile.py -R " + str(data_experiments[experiment]["ring_size"]) + " secure_dual_ls_svm"
    print("Running:", compile_command)
    result = subprocess.run([compile_command], stdout=subprocess.PIPE, shell=True)
    result.check_returncode()


def save_experiment_output(experiment, output, n_execution):
    file_name = "ouput_secure_" + str(n_execution) + ".txt"
    path = config["experiments_path"] + experiment + "/"
    file_output = open(path + file_name, "w")
    file_output.write(output)
    file_output.close()


def generate_dataset_experiment(experiment, n_execution):
    X, y = dataset_generator.generate_dataset(
        math.ceil(data_experiments[experiment]["n_rows"] / data_experiments[experiment]["train_percentage"]), 
        data_experiments[experiment]["n_columns"], 
        data_experiments[experiment]["class_sep"]
    )

    X_train, X_test, y_train, y_test = dataset_generator.split_dataset(X, y, data_experiments[experiment]["train_percentage"])

    dataset_generator.save_dataset_csv(X_train, y_train, experiment, n_execution, "train")
    dataset_generator.save_dataset_csv(X_test, y_test, experiment, n_execution, "test")
    dataset_generator.save_dataset_csv(X, y, experiment, n_execution, "complete")

    dataset_generator.save_dataset_parties(X_train, y_train, data_experiments[experiment]["n_parties"], experiment)


if __name__ == "__main__":
    experiment_list = [
        "test-40r-2c",
        # "test-50r-2c",
        # "test-60r-2c",
        # "test-70r-2c",
        # "test-80r-2c",
        # "test-90r-2c",

        # "test-100r-2c",
        # "test-100r-3c",
        # "test-100r-4c",
        # "test-100r-5c",
        # "test-100r-6c",
        # "test-100r-7c",
        # "test-100r-8c",
        # "test-100r-9c",
        # "test-100r-10c"
    ]

    for experiment in experiment_list:
        execute_experiment(experiment)

    