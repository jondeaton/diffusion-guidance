
import numpy as np

import tokenizers
import tokenizers.models
import tokenizers.pre_tokenizers
import tokenizers.processors

import slip.potts_model

PDB_MSAs = ['3er7', '3bfo', '3gfb', '5hu4', '3my2']


def _tokenizer() -> tokenizers.Tokenizer:
    """Creates tokenizer."""

    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {
                token: i
                for i, token in enumerate(
                    list("LAGVSERTIDPKQNFYMHWCXBUZO") +
                    list("?<>._-")
                )
            },
            unk_token="?",
        )
    )

    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Split(
        tokenizers.Regex("[A-Z]"), behavior="removed", invert=True)

    tokenizer.add_special_tokens(list("?<>._-"))
    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single="$A",  # < $A >
        pair=None,
        special_tokens=[("<", 26), (">", 27)],
    )
    return tokenizer


class Landscape:

    def __init__(
        self, pdb: str,
        coupling_scale: float = 1.0,
        measurement_noise: float = 0.0,
    ):
        self.pdb = pdb
        self.measurement_noise = measurement_noise
        self.potts = slip.potts_model.load_from_mogwai_npz(
            f"slip/data/{pdb}_1_A_model_state_dict.npz",
            coupling_scale=coupling_scale,
        )
        self.tokenizer = _tokenizer()
        self.vocab = list("LAGVSERTIDPKQNFYMHWCXBUZO")

    @property
    def wildtype(self) -> str:
        return self.potts.wildtype_sequence

    def fitness(self, sequence: str) -> float:
        ids: list[int] = self.tokenizer.encode(sequence).ids
        return self.potts.evaluate(ids)

    def batch_fitness(self, sequences: list[str]) -> np.ndarray:
        ids = self.tokenizer.encode_batch(sequences).ids
        return self.potts.evaluate(ids)

    def measure(self, sequences: list[str]) -> np.ndarray:
        fitness = self.batch_fitness(sequences)
        noise = np.random.normal(size=len(sequences))
        return fitness + self.measurement_noise * noise

