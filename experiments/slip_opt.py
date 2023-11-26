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
        measurement_noise=0.1,
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

    df = pd.DataFrame({"sequence": [landscape.wildtype]})
    df['measurement'] = df.sequence.apply(landscape.measure)

    for round in range(config.num_rounds):
        print(f'Round: {round}')

        df['split'] = random.choices(
            ['train', 'test'],
            k=df.shape[0],
            weights=[0.8, 0.2]
        )
        print(f'split sizes: {df.split.value_counts()}')

        model = xgb.Model()
        model.fit(
            df[df.split == "train"].sequence,
            df[df.split == "train"].measurement,
        )

        df_batch = design.design_batch(
            acquisition_fn=model,
            sequences=df[df.split == "test"].sequence,
            batch_size=config.batch_size,
        )
        df_batch['round'] = round
        df_batch['measurement'] = df_batch.sequence.apply(landscape.measure)
        df = pd.concat([df, df_batch], axis=0)

        # Round metrics.
        df['fitness'] = df.sequence.apply(landscape.fitness)
        df = df.sort_values(by='fitness', ascending=False, inplace=True)
        if not DEBUG:
            wandb.log({
                'round': round,
                'round_avg': df[df.round == round].fitness.mean(),
                'top': df.fitness.max(),
                'top10': df.head(10).fitness.min(),
                'avg': df.fitness.mean(),
            })

    if not DEBUG: wandb.finish()


if __name__ == "__main__":
    main()
