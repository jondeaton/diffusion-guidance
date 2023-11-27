
import numpy as np
import pandas as pd

from tqdm import tqdm
import random

from typing import Callable


def design_batch(
    acquisition_fn: Callable[[list[str]], list[float]],
    sequences: list[str],
    batch_size: int,
    vocab: list[str],
    pool_size: int | None = None,
    iters: int = 10,
):
    """Design a batch of sequences with acquisition function."""
    pool_size = pool_size or batch_size
    df = pd.DataFrame({'sequence': sequences})

    pbar = tqdm(range(iters))
    for _ in pbar:
        pool = df.sequence.tolist()
        crosses = [
            crossover(a, b) for a, b in
            zip(
                df.sequence.sample(pool_size, replace=True),
                df.sequence.sample(pool_size, replace=True)
            )
        ]
        pool.extend(crosses)

        muts = [mutate(s, 1, 10, vocab) for s in pool]
        pool.extend(muts)

        df = pd.DataFrame({"sequence": pool})
        df.drop_duplicates('sequence', inplace=True)
        df['af'] = acquisition_fn(df.sequence)
        df = df.sort_values(by='af', ascending=False).head(pool_size)
        pbar.set_description(f'af: {df.af.mean()}')

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
) -> str:
    chars = bytearray(seq, 'utf8')
    num_mutations = np.random.randint(min_mutations, max_mutations + 1)
    positions = random.choices(range(len(seq)), k=num_mutations)
    for pos in positions:
        chars[pos] = ord(random.choice(vocab))
    return chars.decode()

