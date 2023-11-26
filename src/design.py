
import numpy as np
import pandas as pd

from tqdm import tqdm
import random

def design_batch(
    model: Callable[[list[str]], np.ndarray],
    seqs: list[str],
    batch_size: int,
    iters: int = 10,
):
    df = pd.DataFrame({'sequence': seqs})

    pbar = tqdm(range(iters))
    for _ in pbar:
        df.drop_duplicates('sequence')

        df['pred'] = model(df.sequence)
        avg = df.pred.mean()
        pbar.set_description(f"average score: {avg:.4f}")
        df = df.sort_values(by='pred', ascending=False).head(100)

        seqs = []
        for seq in df.seq:

            muts = sampling.sample_within_hamming_radius(
                seq, 10,
                landscape.vocab_size,
                min_mutations=1,
                max_mutations=3,
            )
            seqs.extend(muts)
        df = pd.DataFrame({'sequence': seqs})

    tokens = torch.tensor(df.seq, dtype=torch.long).to(device)
    df['pred'] = model.forward(tokens)[:, 0].cpu().detach().numpy()
    return df.sort_values(by='pred', ascending=False).head(batch_size)


def crossover(s0: str, s1: str):
    assert len(s0) == len(s1)
    i = random.randint(0, len(s0) - 1)
    if random.randint(0, 1):
        s0, s1 = s1, s0
    return s0[:i] + s1[i:]

def mutate(seq: str, num_mutations: int):
    chars = list(seq)
    positions = np.random.randint(0, len(seq))
    for i in positions:
        chars[i] = random.choice

