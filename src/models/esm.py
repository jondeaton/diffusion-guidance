
import torch
import esm

import numpy as np
import torch.nn as nn
from tqdm import tqdm


def mlp(in_size: int, hidden_size: int, out_size: int):
    return nn.Sequential(
        nn.LayerNorm(in_size),
        nn.Linear(in_size, hidden_size),
        nn.GELU(),
        nn.Dropout(p=0.1),
        nn.Linear(hidden_size, out_size)
    )


class AttentionPooling(nn.Module):

    def __init__(
        self,
        in_size: int,
        d_pooled: int = 128,
        hidden_size: int = 128,
        n_regressors: int = 16,
    ):
        super().__init__()
        self.attn = mlp(in_size, hidden_size, 1)
        self.attn_drop = nn.Dropout(0.1)

        self.v = mlp(in_size, hidden_size, d_pooled)

        self.outs = nn.ModuleList([
            mlp(d_pooled, hidden_size, 2)
            for _ in range(n_regressors)
        ])

    def forward(self, x):
        a = torch.exp(self.attn(x)[..., 0])
        a = self.attn_drop(a) # todo: do in log space
        a /= a.sum(axis=1, keepdims=True)

        v = self.v(x)

        z = torch.sum(v * a[..., None], axis=1)

        outs = torch.concat(
            [out(z)[:, None, :] for out in self.outs],
            axis=1
        )

        mus = outs[:, :, 0]
        log_stds = outs[:, :, 1]
        return mus, log_stds


def gaussian_nll(mus, log_stds, y):
    vars = torch.exp(2 * log_stds)
    nll = torch.square(mus - y[:, None]) / (2 * vars) + log_stds
    return nll.mean()


class Model:

    def __init__(self):
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        self.batch_converter = alphabet.get_batch_converter()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        d_model = model.emb_layer_norm_after.normalized_shape[0]

        self.pooling = AttentionPooling(in_size=d_model).to(self.device)
        self.pooling.train()

    def fit(self, seqs, y):
        tokens = self.tokenize(seqs)
        x = self._get_reprs(tokens)
        y = torch.tensor(y, device=self.device)
        # tokens_val = self.batch_converter(seqs_val)
        # x_val = self._get_reprs(tokens_val)

        optimizer = torch.optim.AdamW(
            self.pooling.parameters(), lr=0.001, weight_decay=0.1)
        
        pbar = tqdm(range(100))
        for i in pbar:
            optimizer.zero_grad()
            mus, log_stds = self.pooling(x)

            loss = gaussian_nll(mus, log_stds, y)

            loss.backward()
            optimizer.step()

            loss_val = loss.to('cpu').detach().numpy()
            pbar.set_description(f'loss: {loss_val:.5f}')

            if i % 100 == 0:
                ...
                # train_pred = preds.to('cpu').detach().numpy()
                # test_pred =  self.pooling(x_val).to('cpu').detach().numpy()

                # wandb.log({
                #     'loss': loss,
                #     'train_spearman': scipy.stats.spearmanr(y_train.to('cpu').detach().numpy(), train_pred).correlation,
                #     'test_spearman': scipy.stats.spearmanr(y_test.to('cpu').detach().numpy(), test_pred).correlation,
                #     'train_l2': np.square(y_train.to('cpu').detach().numpy() - train_pred).mean(),
                #     'test_l2': np.square(y_test.to('cpu').detach().numpy() - test_pred).mean(),
                # })
    
    def tokenize(self, sequences: list[str]) -> torch.Tensor:
        _, _, tokens =  self.batch_converter(list(enumerate(sequences)))
        return tokens

    def _get_reprs(self, tokens):
        num_layers = len(self.model.layers)
        # tood: OOM here when batch is too big... maybe need to loop through it
        results = self.model(
            tokens.to(self.device), repr_layers=[num_layers])
        return results['representations'][num_layers]
    
    def predict(self, sequences: list[str], return_std: bool = False) -> tuple[np.ndarray, np.ndarray]:
        tokens = self.tokenize(sequences)
        x = self._get_reprs(tokens)

        mus, log_stds = self.pooling(x)
        mu = mus.mean(axis=1)

        vars = torch.exp(2 * log_stds)
        std = (vars + torch.square(mus)).mean(axis=1) - torch.square(mu)

        mu = mu.to('cpu').detach().numpy()
        std = std.to('cpu').detach().numpy()

        if return_std:
            return mu, std
        else:
            return mu