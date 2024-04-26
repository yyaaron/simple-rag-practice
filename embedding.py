import time
from typing import Union, List

import torch
from numpy import ndarray
from sentence_transformers import SentenceTransformer
from torch import Tensor


class Embedding:
    model_name = ""
    device = ""
    embedding_model = None

    def __init__(self, model_name="all-mpnet-base-v2", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.device = device
        self.embedding_model = SentenceTransformer(model_name_or_path=model_name, device=device)

    def encode(self, sentences: str | list[str], batch_size: int = 32,
               convert_to_tensor: bool = False) -> Union[List[Tensor], ndarray, Tensor]:
        embeddings = self.embedding_model.encode(sentences, batch_size=batch_size,
                                                 convert_to_tensor=convert_to_tensor, device=self.device)

        return embeddings
