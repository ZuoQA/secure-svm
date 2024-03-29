from numpy.lib.npyio import load
from sklearn import datasets
import numpy as np
import flp_svm
import flp_dual_svm
import flp_dual_svm_simp
import flp_dual_svm_fast
import flp_dual_svm_mix
import flp_dual_svm_ls
import flp_dual_svm_gs
import pandas as pd
import matplotlib.pyplot as plt
import datetime

def generate_dataset(n_samples, n_features):
    X, y = datasets.make_classification(n_samples, n_features, n_redundant=0, n_informative=2)
    y = pd.Series(y).map({0: -1, 1: 1}).values

    # Extend y columns
    y = np.expand_dims(y, axis=1)

    # Save dataset for MATLAB testing
    df_save = pd.DataFrame(data=np.append(X, y, axis=1))
    df_save.to_csv("source/datasets/toy_dataset.csv", index=False, columns=None)

    return X, y

def load_dataset(filename):
    df = pd.read_csv("source/datasets/" + filename)
    X = df.iloc[:, :df.shape[1] - 1]
    y = df.iloc[:, df.shape[1] - 1]

    y = np.expand_dims(y, axis=1)

    return X.to_numpy(), y

X, y = load_dataset("toy_dataset_train.csv")

# Print shape of dataset
print("X shape =", X.shape)
print("y shape =", y.shape)

svm = flp_svm.FlpSVM(C=4, lr=0.01)
time_a = datetime.datetime.now()
svm.fit(X, y, epochs=30, verbose=0)
print("Fit time linear =", datetime.datetime.now() - time_a)
training_score = svm.score(X, y)
print("Accuracy linear =", training_score)
print(svm.W)
# print("------------------------------")

# svm_ls = flp_dual_svm_ls.FlpDualLSSVM(lambd=1, lr=0.1, max_iter=50, tolerance=1e-3, kernel="linear")
# time_a = datetime.datetime.now()
# svm_ls.fit(X, y)
# print("Fit time LS =", datetime.datetime.now() - time_a)
# training_score = svm_ls.score(X, y)
# print("Accuracy LS =", training_score)
# print(svm_ls.alphas)
# print("Steps =", svm_ls.steps)

# print("------------------------------")

# svm_gs = flp_dual_svm_gs.FlpDualGSSVM(lambd=4)
# time_a = datetime.datetime.now()
# svm_gs.fit(X, y)
# print(svm_gs.alphas)
# print("Fit time GS =", datetime.datetime.now() - time_a)
# training_score = svm_gs.score(X, y)
# print("Accuracy GS =", training_score)

# print("------------------------------")
# svm_dual = flp_dual_svm.FlpDualSVM(C=1)
# time_a = datetime.datetime.now()
# svm_dual.fit(X, y)
# print("Fit time dual =", datetime.datetime.now() - time_a)
# training_score = svm_dual.score(X, y)
# print("Accuracy dual =", training_score)
# print("Steps =", svm_dual.steps)

# print("------------------------------")

# svm_dual_simp = flp_dual_svm_simp.FlpDualSVMSimp(C=4)
# time_a = datetime.datetime.now()
# svm_dual_simp.fit(X, y)
# print(svm_dual_simp.alphas)
# print("Fit time dual simp =", datetime.datetime.now() - time_a)
# training_score_simp = svm_dual_simp.score(X, y)
# print("Accuracy dual simp =", training_score_simp)
# print("Steps =", svm_dual_simp.steps)

# print("------------------------------")

# svm_dual_mix = flp_dual_svm_fast.SVM(C=4)
# time_a = datetime.datetime.now()
# y_new = np.concatenate(y)
# svm_dual_mix.fit(X, y_new)
# print("Fit time dual fast =", datetime.datetime.now() - time_a)
# training_score_simp = svm_dual_mix.score(X, y_new)
# print("Accuracy dual fast =", training_score_simp)
# print("Steps =", svm_dual_mix.steps)

prediction = svm.predict(X)

fig, axs = plt.subplots(2, 2)

# axs[0, 0].plot(svm.info["accuracy"])
# axs[0, 0].set_title("Accuracy")

# axs[0, 1].plot(svm.info["pk_norm"], color='blue', lw=2)
# axs[0, 1].set_yscale("log")
# axs[0, 1].set_title("Pk norm")

axs[0, 0].scatter(X[:,0], X[:,1], c=y, cmap='viridis')
axs[0, 0].set_title("Real dataset")

axs[0, 1].scatter(X[:,0], X[:,1], c=prediction, cmap='viridis')
axs[0, 1].set_title("Predictions")

plt.show() 
