
import numpy as np
import pandas as pd

from tqdm import tqdm
import random

from typing import Callable


def design_batch(
    acquisition_fn: Callable[[list[str]], list[float]],
    sequences: list[str],
    batch_size: int,
    pool_size: int | None = None,
    iters: int = 4,
):
    """Design a batch of sequences with acquisition function."""
    pool_size = pool_size or batch_size
    df = pd.DataFrame({'sequence': sequences})

    for _ in range(iters):
        pool = set(df.sequence)
        cross = [
            crossover(a, b) for a, b in
            zip(df.sequence.sample(pool_size), df.sequence.sample(pool_size))
        ]
        pool.union(cross)

        for sequence in pool:
            pool.union(mutate(sequence, 1, 3, vocab))

        df = pd.DataFrame({"sequence": pool})
        df['af'] = acquisition_fn(df.sequence)
        df = df.sort_values(by='af', ascending=False).head(pool_size)

    return df.sort_values(by='af', ascending=False).head(batch_size)


def design_batches(
    batch_af: Callable[[list[str]], float],
):
    del batch_af
    # TODO: batch BO design.
    ...


def crossover(s0: str, s1: str):
    """Combines two sequences with crossover."""
    assert len(s0) == len(s1), "Sequences must be same length."
    i = random.randint(0, len(s0) - 1)
    if random.randint(0, 1):
        s0, s1 = s1, s0
    return s0[:i] + s1[i:]


def mutate(
    seq: str,
    min_mutations: int,
    max_mutations: int,
    vocab: list[str],
) -> list[str]:
    chars = list(seq)
    positions = np.random.randint(0, len(seq))
    for i in positions:
        chars[i] = random.choice

