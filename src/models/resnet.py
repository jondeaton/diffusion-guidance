
import jax
import jax.numpy as jnp

import equinox as eqx
from jaxtyping import Float, Integer, Array


class ResidualBlock(eqx.Module):

    convs: list[eqx.nn.Conv]
    layer_norm: eqx.nn.LayerNorm

    def __init__(
            self, num_channels: int, key,
            hidden_channels: int | None = None):
        keys = jax.random.split(key)
        hidden_channels = hidden_channels or num_channels
        self.convs = [
            eqx.nn.Conv(
                num_spatial_dims=1,
                in_channels=num_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                key=keys[i],
            )
            for i, kernel_size in enumerate([1, 3, 5, 7])
        ]
        self.layer_norm0 = eqx.nn.LayerNorm(shape=(,))

        self.conv_out = eqx.nn.Conv(
            num_spatial_dims=1,
            in_channels=4 * hidden_channels,
            out_channels=num_channels,
            kernel_size=1,
            key=key,
        )
        self.layer_norm1 = eqx.nn.LayerNorm(shape=(...,))

    def __call__(self, x: Float[Array, "L C"]):
        x_ = jnp.concatenate([conv(x) for conv in self.convs], axis=1)
        x_ = self.layer_norm(x_)
        x_ = jax.nn.gelu(x_)

        x_ = jax.nn.gelu(self.layer_norm1(self.conv_out(x_)))
        return x + x_


class ResNet(eqx.Module):
    embed: eqx.nn.Embedding
    blocks: list[ResidualBlock]
    head: eqx.nn.MLP

    def __init__(self, num_blocks: int, vocab_size: int, d_model: int, key):
        self.vocab_size = vocab_size
        self.embed = eqx.nn.Embedding(vocab_size, d_model, key)
        keys = jax.random.split(key, num_blocks)
        self.blocks = [
            ResidualBlock(d_model, key=keys[i]) for i in range(num_blocks)
        ]
        self.head = eqx.nn.MLP(
            in_size=d_model,
            out_size=1,
            width_size=d_model,
            depth=1,
            activation=jax.nn.gelu,
            key=key,
        )


    def __call__(self, tokens: Integer[Array, "L"]):
        x = jax.nn.one_hot(tokens, num_classes=self.vocab_size)
        for block in self.blocks:
            x = block(x)
        return self.head(x.mean(axis=0))[:, 0]


class Model():

    def __init__(self):
        self.tokenizer = Tokenizer(...)
        self.model = ResNet(...)
        self._fwd = eqx.filter_jit(eqx.filter_vmap(self.model))

    def __call__(self, sequences: list[str]):
        tokens = self.tokenizer.encode_batch(sequences)
        return self._fwd(tokens)




def fit(
    seqs_train, y_train,
    seqs_test, y_test
):
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


