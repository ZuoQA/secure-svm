import subprocess
import json
import math
import numpy

import dataset_generator


with open("config.json") as config_file:
    config = json.load(config_file)


def execute_experiment(experiment, n_times):
    first_dataset = False
    n_first_execution = 0
    for n_execution in range(n_times):
        if not first_dataset or config["change_dataset"]:
            generate_dataset_experiment(experiment, n_execution)
            first_dataset = True
            n_first_execution = n_execution

        # Experiment subprocess
        if not config["change_dataset"]:
            command = config["experiments_path"] + experiment + "/" + config["mp_spdz_path"] + config["sec_execution_command"]
            result = subprocess.run([command], stdout=subprocess.PIPE)
            result_str = result.stdout.decode('utf-8')
            save_experiment_output(experiment, result_str, n_first_execution)
        else:
            command = config["experiments_path"] + experiment + "/" + config["mp_spdz_path"] + config["sec_execution_command"]
            result = subprocess.run([command], stdout=subprocess.PIPE)
            result_str = result.stdout.decode('utf-8')
            save_experiment_output(experiment, result_str, n_execution)


def save_experiment_output(experiment, output, n_execution):
    pass


def generate_dataset_experiment(experiment, n_execution):
    with open("experiment_info.json") as file_info:
        data_experiments = json.load(file_info)

    X, y = dataset_generator.generate_dataset(
        math.ceil(data_experiments[experiment]["n_rows"] / data_experiments[experiment]["train_percentage"]), 
        data_experiments[experiment]["n_cols"], 
        data_experiments[experiment]["class_sep"]
    )

    X_train, X_test, y_train, y_test = dataset_generator.split_dataset(X, y, data_experiments[experiment]["train_percentage"])

    # TODO Vinculate the experiment and the execution numbers into the file names
    dataset_generator.save_dataset_csv(X_train, y_train, "train")
    dataset_generator.save_dataset_csv(X_test, y_test, "test")
    dataset_generator.save_dataset_csv(X, y, "complete")

    dataset_generator.save_dataset_parties(X_train, y_train, data_experiments[experiment]["n_parties"])


if __name__ == "__main__":
    
    n_times = 10
    experiment_list = [
        "test-20r-2c"
    ]

    