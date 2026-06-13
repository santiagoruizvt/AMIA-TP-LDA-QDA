import numpy as np
import numpy.linalg as LA
from scipy.linalg import cholesky, solve_triangular
from scipy.linalg.lapack import dtrtri

from base.bayesian import BaseBayesianClassifier


class QDA_Chol1(BaseBayesianClassifier):
  def _fit_params(self, X, y):
    self.L_invs = [
        LA.inv(cholesky(np.cov(X[:,y.flatten()==idx], bias=True), lower=True))
        for idx in range(len(self.log_a_priori))
    ]

    self.means = [X[:,y.flatten()==idx].mean(axis=1, keepdims=True)
                  for idx in range(len(self.log_a_priori))]

  def _predict_log_conditional(self, x, class_idx):
    L_inv = self.L_invs[class_idx]
    unbiased_x =  x - self.means[class_idx]

    y = L_inv @ unbiased_x

    return np.log(L_inv.diagonal().prod()) -0.5 * (y**2).sum()


class QDA_Chol2(BaseBayesianClassifier):
  def _fit_params(self, X, y):
    self.Ls = [
        cholesky(np.cov(X[:,y.flatten()==idx], bias=True), lower=True)
        for idx in range(len(self.log_a_priori))
    ]

    self.means = [X[:,y.flatten()==idx].mean(axis=1, keepdims=True)
                  for idx in range(len(self.log_a_priori))]

  def _predict_log_conditional(self, x, class_idx):
    L = self.Ls[class_idx]
    unbiased_x =  x - self.means[class_idx]

    y = solve_triangular(L, unbiased_x, lower=True)

    return -np.log(L.diagonal().prod()) -0.5 * (y**2).sum()


class QDA_Chol3(BaseBayesianClassifier):
  def _fit_params(self, X, y):
    self.L_invs = [
        dtrtri(cholesky(np.cov(X[:,y.flatten()==idx], bias=True), lower=True), lower=1)[0]
        for idx in range(len(self.log_a_priori))
    ]

    self.means = [X[:,y.flatten()==idx].mean(axis=1, keepdims=True)
                  for idx in range(len(self.log_a_priori))]

  def _predict_log_conditional(self, x, class_idx):
    L_inv = self.L_invs[class_idx]
    unbiased_x =  x - self.means[class_idx]

    y = L_inv @ unbiased_x

    return np.log(L_inv.diagonal().prod()) -0.5 * (y**2).sum()
  
class TensorizedChol(QDA_Chol3):

    def _fit_params(self, X, y):
        super()._fit_params(X, y)
        # apilar L_invs y means en tensores
        self.tensor_L_inv = np.stack(self.L_invs)   # (k, p, p)
        self.tensor_means = np.stack(self.means)     # (k, p, 1)

    def _predict_log_conditionals(self, x):
        # x: (p, 1)
        unbiased_x = x - self.tensor_means           # (k, p, 1)

        # y = L⁻¹(x - μ) para cada clase: (k, p, p) @ (k, p, 1) = (k, p, 1)
        y = self.tensor_L_inv @ unbiased_x           # (k, p, 1)

        # ||y||² para cada clase: (k,)
        quad = (y**2).sum(axis=(1, 2))               # (k,)

        # log det(L⁻¹) = sum(log diagonal) para cada clase: (k,)
        log_det = np.log(self.tensor_L_inv[:, np.arange(y.shape[1]),
                                            np.arange(y.shape[1])]).sum(axis=1)

        return log_det - 0.5 * quad

    def _predict_one(self, x):
        return np.argmax(self.log_a_priori + self._predict_log_conditionals(x))
    
class EfficientChol(TensorizedChol):

    def predict(self, X):
        # X: (p, n)
        unbiased = X[np.newaxis, :, :] - self.tensor_means   # (k, p, n)

        # Y = L⁻¹(X - μ) para cada clase: (k, p, p) @ (k, p, n) = (k, p, n)
        Y = self.tensor_L_inv @ unbiased                      # (k, p, n)

        # ||y_i||² = sum(Y ⊙ Y, axis=p) → (k, n)  — equivalente al producto Hadamard
        quad = np.sum(Y * Y, axis=1)                          # (k, n)

        # log det por clase: (k, 1)
        log_det = np.log(np.diagonal(
            self.tensor_L_inv, axis1=1, axis2=2
        )).sum(axis=1)[:, None]                               # (k, 1)

        log_cond = log_det - 0.5 * quad                       # (k, n)
        scores = self.log_a_priori[:, None] + log_cond        # (k, n)

        return np.argmax(scores, axis=0).reshape(1, -1)
