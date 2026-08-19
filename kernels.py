import numpy as np
from numpy import linalg
import cvxxopt
import cvxopt.solvers

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x1, x2, degree=3):
    return (np.dot(x1, x2) + 1) ** degree

def gussian_kernel(x1, x2, sigma=1.0):
    return np.exp(-linalg.norm(x1 - x2) ** 2 / (2 * (sigma ** 2)))

class SVM:
    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None:
            self.C = float(self.C)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-np.ones(n_samples))
        A = cvxopt.matrix(y, (1, n_samples), 'd')
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(-np.eye(n_samples))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            G = cvxopt.matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
            h = cvxopt.matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        a = np.ravel(solution['x'])

        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv_X = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
        self.b /= len(self.a)

        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv_X[n]
        
        else:
            self.w = None
    
    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv_X):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b
    
    def predict(self, X):
        return np.sign(self.project(X))

if __name__ == "__main__":
    import pylab as pl

    def gen_lin_separable_data():
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[0.8, 0.6], [0.6, 0.8]])
        X1 = np.random.multivariate_normal(mean1, cov, 20)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 20)
        y2 = -np.ones(len(X2))
        return X1, y1, X2, y2

    def gen_non_lin_separable_data():
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        mean3 = np.array([0, -2])
        mean4 = np.array([-2, 0])
        cov = np.array([[0.8, 0.6], [0.6, 0.8]])
        X1 = np.random.multivariate_normal(mean1, cov, 20)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 20)
        y2 = -np.ones(len(X2))
        X3 = np.random.multivariate_normal(mean3, cov, 20)
        y3 = np.ones(len(X3))
        X4 = np.random.multivariate_normal(mean4, cov, 20)
        y4 = -np.ones(len(X4))
        return X1, y1, X2, y2, X3, y3, X4, y4
    
    def gen_lin_separable_overlap_data():
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[1.5, 1.0], [1.0, 1.5]])
        X1 = np.random.multivariate_normal(mean1, cov, 20)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 20)
        y2 = -np.ones(len(X2))
        return X1, y1, X2, y2
    
    def split_train(X1,y1,X2,y2):
        X1_train =  X1[:90]
        y1_train =  y1[:90]
        X2_train =  X2[:90]
        y2_train =  y2[:90]
        x_test = np.vstack((X1_test,X2_test))
        y_test = np.hstack((y1_test,y2_test))
        return X_test, y_test
    
    # def plot_margin(X1_train,X2_train, clf):
    #     pass

    
