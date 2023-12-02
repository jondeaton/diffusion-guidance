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
from src.models import gp

import wandb

from src import slip
from src import tokenizers

import hydra
from omegaconf import DictConfig, OmegaConf

DEBUG: bool = True


@hydra.main(version_base=None, config_path="../conf")
def main(config : DictConfig):

    landscape = slip.Landscape(
        pdb=config.task.slip.pdb,
        coupling_scale=config.task.slip.coupling_scale,
        measurement_noise=config.task.slip.measurement_noise,
    )

    if not DEBUG:
        wandb.init(
            project="slip-design",
            config=config,
        )

    vocab = tokenizers.MOGWAI_VOCAB

    df = pd.DataFrame(
        {
            "sequence": [
                landscape.wildtype,
                *[
                    design.mutate(landscape.wildtype, 3, 5, vocab)
                    for _ in range(config.task.batch_size)
                ]
            ]
        }
    )
    df['measurement'] = landscape.batch_measure(df.sequence)
    df['fitness'] = landscape.batch_fitness(df.sequence)
    print('Starting pool Avg fitness:', df.fitness.mean())

    for round in range(config.task.num_rounds):
        print()
        print(f'Round: {round}')

        df['split'] = random.choices(
            ['train', 'test'],
            k=df.shape[0],
            weights=[0.8, 0.2]
        )
        print(f'split sizes: {df.split.value_counts().tolist()}')

        # model = xgb.Model()
        model = gp.Model(noise_std=config.task.slip.measurement_noise)
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

        f_best = df.measurement.max()
        from scipy.stats import norm

        def acquisition_fn(seqs: list[str]) -> np.ndarray:
            mu, sigma = model.predict(seqs, return_std=True)

            # Upper Confidence Bound.
            return mu + (1 / (1 + round)) * sigma

            # Expected Improvement.
            # Z = (mu - f_best) / sigma
            # return (mu - f_best) * norm.cdf(Z) + sigma * norm.pdf(Z)

        df_batch = design.design_batch(
            acquisition_fn=acquisition_fn,
            sequences=df[df.split == "test"].sequence,
            batch_size=config.task.batch_size,
            pool_size=config.design.pool_size,
            vocab=vocab,
            iters=config.design.iters,
        )

        df_batch['round_'] = round
        df_batch['measurement'] = landscape.batch_measure(df_batch.sequence)
        df_batch['fitness'] = landscape.batch_fitness(df_batch.sequence)
        df = pd.concat([df, df_batch], axis=0)

        # Round metrics.
        df.sort_values(by='fitness', ascending=False, inplace=True)
        round_average = df[df.round_ == round].fitness.mean()
        print(f'Average fitness: {round_average:.4f}')

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
