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

import jax
import jax.numpy as jnp
import equinox as eqx

import wandb

from slip import potts_model, sampling, experiment

import equinox as eqx
import optax
from jaxtyping import Float, Integer, Array, PRNGKeyArray

from experiments import resnet
from src import design


def fit_model(df: pd.DataFrame) -> nn.Module:
    df_train = df[df.split == "train"]
    df_test = df[df.split == "test"]

    tokens_train = np.array(df_train.seq.tolist())
    y_train = df_train.fitness.values

    tokens_test = df_test.seq.tolist()
    y_test = df_test.fitness.values

    model = resnet.ResNet(num_blocks=3, vocab_size=20)
    optimizer = optax.adamw(0.001, weight_decay=0.1)
    opt_state = optimizer.init(model)

    def update(
        model: eqx.Module,
        opt_state: optax.OptState,
        x, y,
    ) -> tuple[Float, eqx.Module, optax.OptState]:

        def loss(model, x, y):
            y_ = jax.vmap(model)(x)
            return optax.huber_loss(y, y_).mean()

        l, grads = eqx.filter_value_and_grad(loss)(model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = optax.apply_updates(model, updates)

        return l, model, opt_state

    for i in tqdm(range(1000)):
        loss, model, opt_state = update(model, opt_state, tokens_train, y_train)

        train_pred = jax.vmap(model)(tokens_train)
        test_pred = jax.vmap(model)(tokens_test)

        wandb.log({
            'loss': loss,
            'train_spearman': tspearmanr(y_train, train_pred).correlation,
            'test_spearman': spearmanr(y_test, test_pred).correlation,
        })

    return model


@dataclasses.dataclass
class Config:
    num_rounds: int
    batch_size: int

    # Landscape.
    pdb: str
    coupling_scale: float
    measurement_noise: float


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

        model = fit_model(df)
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
        wandb.log({
            'round': round,
            'round_avg': df[df.round == round].fitness.mean(),
            'top': df.fitness.max(),
            'top10': df.head(10).fitness.min(),
            'avg': df.fitness.mean(),
        })

    wandb.finish()

if __name__ == "__main__":
    main()
