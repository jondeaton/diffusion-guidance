
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel


def hamming_distance(s0: str, s1: str) -> int:
    assert len(s0) == len(s1)
    return sum(c1 != c2 for c1, c2 in zip(s0, s1))


class HammingDistanceKernel(Kernel):

    def __init__(self, length_scale: float = 1.0):
        self.length_scale = length_scale

    def is_stationary(self):
        return False

    def diag(self, X):
        n, _ = X.shape
        return np.ones(n)

    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
        d = np.sum(X[:, None] != Y[None, :], axis=2)
        return 1 / (1 + d / self.length_scale)


def featurize(seq: str):
    return [ord(c) for c in seq]


class Model:

    def __init__(self, noise_std: float = 0.):
        self.gp = GaussianProcessRegressor(
            kernel=HammingDistanceKernel(length_scale=3),
            alpha=2 * noise_std,
        )

    def fit(self, seqs, y):
        X = np.array([featurize(s) for s in seqs])
        self.gp.fit(X, y)

    def predict(self, seqs, return_std: bool = False):
        X = np.array([featurize(s) for s in seqs])
        return self.gp.predict(X, return_std=return_std)

