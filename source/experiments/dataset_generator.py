from sklearn import datasets
import numpy as np
import pandas as pd
import random


# Set seeds for RNGs
np.random.seed(1)
random.seed(1)


def generate_dataset(n_samples, n_features):
    X, y = datasets.make_classification(n_samples, n_features, n_redundant=0, n_informative=2)
    y = pd.Series(y).map({0: -1, 1: 1}).values

    # Extend y columns
    y = np.expand_dims(y, axis=1)

    df_save = pd.DataFrame(data=np.append(X, y, axis=1))
    df_save.to_csv("source/experiments/toy_dataset.csv", index=False, columns=None)

    return X, y


def save_dataset_parties(X, y, n_rows, n_cols, n_parties):
    rows_per_party = n_rows // n_parties
    last_party = 0 
    if n_rows % n_parties != 0:
        last_party = rows_per_party + 1
    else:
        last_party = rows_per_party
    
    party_info_X = []
    party_info_y = []
    for i in range(n_parties - 1):
        party_X_rows = []
        party_y_rows = []
        for j in range(rows_per_party):
            party_X_rows.append(X[j + i * rows_per_party].tolist())
            party_y_rows.append(y[j + i * rows_per_party][0])
        party_info_X.append(party_X_rows)
        party_info_y.append(party_y_rows)

    # Last party
    party_X_rows = []
    party_y_rows = []
    for j in range(last_party):
        party_X_rows.append(X[j + rows_per_party * (n_parties - 1)].tolist())
        party_y_rows.append(y[j + rows_per_party * (n_parties - 1)][0])
    party_info_X.append(party_X_rows)
    party_info_y.append(party_y_rows)

    for i in range(n_parties - 1):
        file_name = "source/experiments/Input-P" + str(i) + "-0"
        file = open(file_name, "w")
        file_str = ""
        for j in range(rows_per_party):
            for k in range(n_cols):
                file_str += str(party_info_X[i][j][k]) + " "
            file_str = file_str.strip()
            file_str += "\n"
        
        for j in range(rows_per_party):
            file_str += str(party_info_y[i][j]) + "\n"
        
        file.write(file_str)
        file.close()
    
    # Last party write
    file_name = "source/experiments/Input-P" + str(n_parties - 1) + "-0"
    file = open(file_name, "w")
    file_str = ""
    for j in range(last_party):
        for k in range(n_cols):
            file_str += str(party_info_X[n_parties - 1][j][k]) + " "
        file_str = file_str.strip()
        file_str += "\n"
    
    for j in range(last_party):
        file_str += str(party_info_y[n_parties - 1][j]) + "\n"
    
    file.write(file_str)
    file.close()

if __name__ == "__main__":
    N_ROWS = 80
    N_COLS = 2
    N_PARTIES = 4

    X, y = generate_dataset(N_ROWS, N_COLS)
    save_dataset_parties(X, y, N_ROWS, N_COLS, N_PARTIES)