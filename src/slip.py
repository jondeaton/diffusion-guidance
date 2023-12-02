
import numpy as np
import slip.potts_model
from src import tokenizers


PDB_MSAs = ['3er7', '3bfo', '3gfb', '5hu4', '3my2']


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
        self.tokenizer = tokenizers.basic_tokenizer()
        self.vocab = tokenizers.MOGWAI_VOCAB

    @property
    def wildtype(self) -> str:
        return self.tokenizer.decode(self.potts.wildtype_sequence)

    def fitness(self, sequence: str) -> float:
        ids: list[int] = self.tokenizer.encode(sequence).ids
        return self.potts.evaluate(ids)[0]

    def batch_fitness(self, sequences: list[str]) -> np.ndarray:
        ids = [t.ids for t in self.tokenizer.encode_batch(sequences)]
        return self.potts.evaluate(ids)

    def measure(self, sequence: str) -> float:
        noise = self.measurement_noise * np.random.normal()
        return self.fitness(sequence) + noise

    def batch_measure(self, sequences: list[str]) -> np.ndarray:
        fitness = self.batch_fitness(sequences)
        noise = np.random.normal(size=len(sequences))
        return fitness + self.measurement_noise * noise

