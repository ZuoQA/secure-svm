from flp_dual_svm_ls import FlpDualLSSVM
import numpy as np

svm = FlpDualLSSVM(kernel="r", gamma=0.4)
a = np.array([[1], [2]])
b = np.array([[1], [2]])
print(svm.kernel(a, b))