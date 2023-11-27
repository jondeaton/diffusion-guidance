"""Experiments optimizing SLIP landscapes with ML. """

import os
import random
import dataclasses
import functools
import collections

from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src import design
from src.models import xgb

import wandb

from src import slip
from src import tokenizers


@dataclasses.dataclass
class Config:
    num_rounds: int
    batch_size: int

    # Landscape.
    pdb: str
    coupling_scale: float
    measurement_noise: float


DEBUG: bool = True


def main():
    config = Config(
        num_rounds=10,
        batch_size=96,
        pdb="3er7",
        coupling_scale=1.0,
        measurement_noise=0.0,
    )

    landscape = slip.Landscape(
        pdb=config.pdb,
        coupling_scale=config.coupling_scale,
        measurement_noise=config.measurement_noise,
    )

    if not DEBUG:
        wandb.init(
            project="slip-design",
            config=dataclasses.asdict(config),
        )

    vocab = tokenizers.MOGWAI_VOCAB

    df = pd.DataFrame(
        {
            "sequence": [
                # landscape.wildtype,
                *[
                    design.mutate(landscape.wildtype, 3, 5, vocab)
                    for _ in range(96)
                ]
            ]
        }
    )
    df['measurement'] = df.sequence.apply(landscape.measure)

    for round in range(config.num_rounds):
        print()
        print(f'Round: {round}')

        df['split'] = random.choices(
            ['train', 'test'],
            k=df.shape[0],
            weights=[0.8, 0.2]
        )
        print(f'split sizes: {df.split.value_counts().tolist()}')

        model = xgb.Model()
        model.fit(
            df[df.split == "train"].sequence,
            df[df.split == "train"].measurement,
        )

        train_spearman = spearmanr(
            model.predict(df[df.split == "train"].sequence),
            df[df.split == "train"].measurement,
        ).statistic
        test_spearman = spearmanr(
            model.predict(df[df.split == "test"].sequence),
            df[df.split == "test"].measurement,
        ).statistic
        print(f'Spearman: train {train_spearman:.4f}, {test_spearman:.4f}')

        df_batch = design.design_batch(
            acquisition_fn=model.predict,
            sequences=df[df.split == "test"].sequence,
            batch_size=config.batch_size,
            pool_size=128,
            vocab=vocab,
            iters=round + 1,
        )
        print(f'AF mean: {df_batch.af.mean()}')

        df_batch['round_'] = round
        df_batch['measurement'] = df_batch.sequence.apply(landscape.measure)
        df = pd.concat([df, df_batch], axis=0)

        # Round metrics.
        df['fitness'] = df.sequence.apply(landscape.fitness)
        df.sort_values(by='fitness', ascending=False, inplace=True)
        round_average = df[df.round_ == round].fitness.mean()
        print(f'Average fitness:', round_average)

        if not DEBUG:
            wandb.log({
                'round_': round,
                'round_avg': round_average,
                'top': df.fitness.max(),
                'top10': df.head(10).fitness.min(),
                'avg': df.fitness.mean(),
            })

    if not DEBUG: wandb.finish()


if __name__ == "__main__":
    main()
