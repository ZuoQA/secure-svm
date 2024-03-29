import numpy as np
import datetime

class FlpDualLSSVM(object):

    def __init__(self, lambd=4, lr=1e-1, max_iter=50, kernel="linear", tolerance=1e-7, degree=None, gamma=None, r=None) -> None:
        super().__init__()
        self.lr = lr
        self.degree = degree
        self.lambd = lambd
        self.tolerance = tolerance
        self.kernel_type = kernel
        self.max_iter = max_iter
        self.gamma = gamma
        self.r = r

    def kernel(self, a, b):
        if self.kernel_type == "linear":
            return a.T.dot(b)[0][0]
        if self.kernel_type == "poly":
            return np.power(1 + a.T.dot(b)[0][0], self.degree)
        if self.kernel_type == "r":
            return np.exp(-self.gamma * np.linalg.norm(a - b) ** 2)
        if self.kernel_type == "sigmoidal":
            return np.tanh(self.gamma * a.T.dot(b)[0][0] + self.r)

    def compute_omega(self):
        omega = np.zeros(shape=(self.data.shape[0], self.data.shape[0]))
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[0]):
                Xi = np.expand_dims(self.data[i], axis=1)
                Xj = np.expand_dims(self.data[j], axis=1)
                omega[i][j] = self.y[i][0] * self.y[j][0] * self.kernel(Xi, Xj)
        return omega
    
    def predict_distance_vect(self, x):
        prediction = 0
        for i in range(self.data.shape[0]):
            Xi = np.expand_dims(self.data[i], axis=1)
            prediction += self.alphas[i][0] * self.y[i][0] * self.kernel(Xi, x)
        
        prediction += self.b

        return prediction

    def predict_distance(self, X):
        predictions = np.zeros(shape=(X.shape[0], 1))
        for i in range(X.shape[0]):
            Xi = np.expand_dims(X[i], axis=1)
            predictions[i][0] = self.predict_distance_vect(Xi)

        return predictions

    def predict(self, X):
        predictions = self.predict_distance(X)
        return np.sign(predictions)

    def compute_A(self, omega, y):
        omega_lamba_id = omega + self.lambd * np.identity(self.data.shape[0])
        
        upper_A = np.concatenate((np.array([[0]]), y.T), axis=1)
        lower_A = np.concatenate((y, omega_lamba_id), axis=1)

        A = np.concatenate((upper_A, lower_A), axis=0)

        return A

    def fit(self, X, y):
        self.data = X
        self.y = y

        self.info = dict()
        self.info["accuracy"] = list()
        self.info["pk_norm"] = list()
        
        self.steps = 0
        
        omega = self.compute_omega()

        A = self.compute_A(omega, y)

        opt_matrix = np.dot(A.T, A)
        ones_hat = np.concatenate((np.array([[0]]), np.ones(shape=(self.data.shape[0], 1))), axis=0)
        opt_vect = np.dot(A.T, ones_hat)

        beta_k = np.random.random(size=(self.data.shape[0] + 1, 1))
        for i in range(self.max_iter):
            p_k = opt_vect - np.dot(opt_matrix, beta_k)
            r_k = np.dot(p_k.T, p_k) / np.dot(p_k.T, np.dot(opt_matrix, p_k))
            
            beta_k = beta_k + (1 - self.lr) * r_k * p_k
            
            self.alphas = beta_k[1:]
            self.b = beta_k[0][0]

            self.info["accuracy"].append(self.score(self.data, self.y))
            self.info["pk_norm"].append(np.linalg.norm(p_k))
            
            #print("||pk|| =", np.linalg.norm(p_k))
            if np.linalg.norm(p_k) < self.tolerance:
                break
            
            self.steps += 1
            
        self.alphas = beta_k[1:]
        self.b = beta_k[0][0]

        return self.alphas, self.b

    def score(self, X, y_true):
        predict = self.predict(X)
        n_correct = np.sum(predict == y_true)
        return n_correct / X.shape[0]

    def load_parameters(self, alphas, b, X_train, y_train):
        self.alphas = alphas
        self.b = b
        self.data = X_train
        self.y = y_train