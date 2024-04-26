import pandas as pd
import numpy as np
import torch


class EmbeddingLoader:
    device = ""
    file_name = ""

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu",
                 file_name="text_chunks_and_embeddings_df.csv"):
        self.device = device
        self.file_name = file_name

    @classmethod
    def load(cls, file_path: str = "text_chunks_and_embeddings_df.csv",
             device="cuda" if torch.cuda.is_available() else "cpu"):
        print(f"[INFO]Load embeddings of texts from file:{file_path}...")
        text_chunks_and_embedding_df = pd.read_csv(file_path)
        text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(
            lambda x: np.fromstring(x.strip("[]"), sep=" "))

        pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")

        embeddings = (torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32)
                      .to(device))
        print(f"[INFO]Finish loading embeddings of texts")
        return pages_and_chunks, embeddings
