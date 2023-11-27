

import xgboost as xgb
import numpy as np
import einops

from src import tokenizers


def one_hot(x: list[int], k: int) -> np.ndarray:
    return np.take(np.eye(k), x, axis=0)


class Model:

    def __init__(self):
        self.model = xgb.XGBRegressor(n_estimators=200)
        self.tokenizer = tokenizers.basic_tokenizer()

    def _featurize(self, seqs: list[str]):
        tokens = [t.ids for t in self.tokenizer.encode_batch(seqs)]
        return einops.rearrange(one_hot(tokens, 20), 'b l v -> b (l v)')

    def fit(self, seqs, y):
        x = self._featurize(seqs)
        self.model.fit(x, y)

    def predict(self, seqs: list[str]) -> list[float]:
        x = self._featurize(seqs)
        return self.model.predict(x)
