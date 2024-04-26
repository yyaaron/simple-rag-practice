import numpy as np
import torch.cuda

import rag_utils
from embedding_loader import EmbeddingLoader
from file_processor import FileDownloader, FilePreprocessor
from embedding import Embedding
from rag_utils import RagUtils
from llm_utils import LlmUtils
import pandas as pd

query = "macronutrients functions"

if __name__ == '__main__':
    # ***********************************
    # STEP 1: File preprocess
    # ***********************************

    # download example PDF file
    file_name = 'human-nutrition-text.pdf'
    download_url = 'https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf'
    FileDownloader().download_if_not_exist(file_name=file_name, download_url=download_url)

    # by default, we use a pdf file named "human-nutrition-text.pdf"
    pages_and_texts = FilePreprocessor.open_and_read_pdf(file_name)

    # turn texts into chunks, and set minimum size of tokens per chunk as 30, as they are always meaningless
    pages_and_trunk_over_min_size = FilePreprocessor.pages_and_chunks(pages_and_texts=pages_and_texts, min_token_len=30)

    # ***********************************
    # STEP 2: Embedding text chunks
    # ***********************************

    # embedding text chunks
    device = "cuda" if torch.cuda.is_available() else "cpu"  # set device "cuda" if gpu available
    embedding_tool = Embedding(model_name="all-mpnet-base-v2", device=device)
    print(f"[INFO]Embedding sentences by model {embedding_tool.model_name}, on device({embedding_tool.device})...")
    for item in pages_and_trunk_over_min_size:
        item["embedding"] = embedding_tool.encode(item["sentence_chunk"], batch_size=32, convert_to_tensor=False)
    print(f"[INFO]Finish embedding.")

    # ***********************************
    # STEP 3: Save embeddings into csv
    # ***********************************

    # embeddings can be saved in any vector database, but here we use csv file for simplicity
    text_chunks_and_embeddings_df = pd.DataFrame(pages_and_trunk_over_min_size)
    embeddings_df_save_path = "text_chunks_and_embeddings_df.csv"
    text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False)

    # ***********************************
    # STEP 4: Load embeddings from CSV
    # ***********************************

    # text_chunks_and_embedding_df = pd.read_csv(embeddings_df_save_path)
    # text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(
    #                                                                 lambda x: np.fromstring(x.strip("[]"), sep=" "))
    # pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")
    # embeddings = (torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32)
    #               .to(device))

    pages_and_chunks, embeddings = EmbeddingLoader.load(file_path=embeddings_df_save_path, device=device)

    # ***********************************
    # STEP 5: Load LLM
    # ***********************************

    # you can select any LLM according to your scenario, here we select google/gemma-2b-it for demo use
    # LLM can be downloaded from huggingface or kaggle.
    # here we download LLM from huggingface, you should have account of it and login to find your token.
    # access token setting page: https://huggingface.co/settings/tokens
    tokenizer, llm = LlmUtils.init_model()

    # ***********************************
    # STEP 6: generate answer
    # ***********************************

    # at last, we ask LLM with query and contexts with the highest relevance.
    # relevance between context and query is computed in RagUtils.retrieve_relevant_resources
    query = "What are the macronutrients, and what roles do they play in the human body?"
    answer = LlmUtils.ask(query=query, embeddings=embeddings, embedding_model=embedding_tool.embedding_model,
                          pages_and_chunks=pages_and_chunks, model=llm, tokenizer=tokenizer,
                          return_answer_only=True)

    print(f"Answer:\n{answer}")

