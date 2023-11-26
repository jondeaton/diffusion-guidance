

import xgboost as xgb
import numpy as np
import einops


def one_hot(x: list[int], k: int) -> np.ndarray:
    return np.take(np.eye(k), x, axis=0)


def featurize(x):
    return einops.rearrange(one_hot(x, 20), 'b l v -> b (l v)')


class Model:

    def __init__(self):
        self.model = xgb.XGBRegressor(n_estimators=100, gamma=10)

    def __call__(self, sequences: list[str]):
        tokens = self.tokenizer()
        feat = featurize(sequences)
        return self.model.predict(train_feat)

    def fit(seqs, y):
        x = featurize(seqs)
        self.model.fit(x, y)
