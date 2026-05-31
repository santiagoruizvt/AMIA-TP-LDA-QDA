import numpy as np
import numpy.linalg as LA

from base.bayesian import BaseBayesianClassifier


class QDA(BaseBayesianClassifier):

  def _fit_params(self, X, y):
    # estimate each covariance matrix
    self.inv_covs = [LA.inv(np.cov(X[:,y.flatten()==idx], bias=True))
                      for idx in range(len(self.log_a_priori))]
    # Q5: por que hace falta el flatten y no se puede directamente X[:,y==idx]?
    # Q6: por que se usa bias=True en vez del default bias=False?
    self.means = [X[:,y.flatten()==idx].mean(axis=1, keepdims=True)
                  for idx in range(len(self.log_a_priori))]
    # Q7: que hace axis=1? por que no axis=0?

  def _predict_log_conditional(self, x, class_idx):
    # predict the log(P(x|G=class_idx)), the log of the conditional probability of x given the class
    # this should depend on the model used
    inv_cov = self.inv_covs[class_idx]
    unbiased_x =  x - self.means[class_idx]
    return 0.5*np.log(LA.det(inv_cov)) -0.5 * unbiased_x.T @ inv_cov @ unbiased_x


class TensorizedQDA(QDA):

    def _fit_params(self, X, y):
        # ask plain QDA to fit params
        super()._fit_params(X,y)

        # stack onto new dimension
        self.tensor_inv_cov = np.stack(self.inv_covs)
        self.tensor_means = np.stack(self.means)

    def _predict_log_conditionals(self,x):
        unbiased_x = x - self.tensor_means
        inner_prod = unbiased_x.transpose(0,2,1) @ self.tensor_inv_cov @ unbiased_x

        return 0.5*np.log(LA.det(self.tensor_inv_cov)) - 0.5 * inner_prod.flatten()

    def _predict_one(self, x):
        # return the class that has maximum a posteriori probability
        return np.argmax(self.log_a_priori + self._predict_log_conditionals(x))


class FasterQDA(TensorizedQDA):
    """Predict multiple observations in a single pass without Python-level loops.

    Uses stacked tensors produced in _fit_params (k, p, p) and (k, p, 1)
    to compute the quadratic form for all classes and all observations
    while avoiding the n x n interaction matrix.
    """

    def predict(self, X):
        # X expected shape: (p, n)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array with shape (p, n)")

        # Build stacked tensors if not already present
        if not hasattr(self, 'tensor_inv_cov') or not hasattr(self, 'tensor_means'):
            # ensure fit was called
            raise RuntimeError("Model parameters not found. Call fit(...) before predict().")

        # unbiased: shape (k, p, n)
        unbiased = X[np.newaxis, :, :] - self.tensor_means

        # transformed = tensor_inv_cov @ unbiased  -> shape (k, p, n)
        transformed = np.einsum('kij,kjn->kin', self.tensor_inv_cov, unbiased)

        # quadratic form per class and observation: sum over p -> shape (k, n)
        quad = np.sum(transformed * unbiased, axis=1)

        # log det per class: shape (k,)
        log_det = 0.5 * np.log(LA.det(self.tensor_inv_cov))

        # log conditional per class x obs: (k, n)
        log_cond = log_det[:, None] - 0.5 * quad

        # add log a priori and pick argmax across classes
        scores = self.log_a_priori[:, None] + log_cond

        preds = np.argmax(scores, axis=0)
        return preds.reshape(1, -1)
