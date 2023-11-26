"""Experiments optimizing SLIP landscapes with ML. """

import os
import random
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



def main():
    pdb = '3er7'  # 3bfo, 3gfb, 5hu4, 3my2

    landscape = potts_model.load_from_mogwai_npz(
        f'slip/data/{pdb}_1_A_model_state_dict.npz', coupling_scale=1.0)

    def fitness(sequences: list[str]) -> list[float]:
        tokens = tokenizer.encode_batch(sequences)
        return landscape.evaluate(tokens)

    wandb.init(
        project="sequence-design",
        config={...},
    )
    sample = sampling.sample_within_hamming_radius(
        landscape.wildtype_sequence, 96,
        landscape.vocab_size,
        1, 10,
    )
    df = pd.DataFrame(
        {'sequence': list(sample), 'fitness': landscape.evaluate(sample)})

    for round in range(num_rounds):
        print(f'round: {round}')

        df['split'] = random.choices(['train', 'test'], k=df.shape[0], weights=[0.8, 0.2])
        print(f'split sizes: {df.split.value_counts()}')

        model = fit_model(df)
        def model_fn(seqs: list[str]) -> np.ndarray:
            tokens = tokenizer.encode_batch(seqs)
            return jax.filter_vmap(model)(tokens)

        df_batch = design.design_batch(
            model_fn,
            df[df.split == "test"].sequence,
            batch_size=96,
            iters=5,
        )

        df_batch['fitness'] = fitness(df_batch.sequence)
        df = pd.concat([df, df_batch], axis=0)
        df = df.sort_values(by='fitness', ascending=False, inplace=True)

        wandb.log({
            'top10avg': df.head(10).fitness.mean(),
            'avg': df.fitness.mean(),
            'round': round,
        })

    wandb.finish()

if __name__ == "__main__":
    main()
