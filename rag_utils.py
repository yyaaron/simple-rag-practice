import torch
from sentence_transformers import util, SentenceTransformer
from time import perf_counter as timer
import textwrap


class RagUtils:
    def __init__(self):
        pass

    @classmethod
    def retrieve_relevant_resources(cls,
                                    query: str,
                                    embeddings: torch.tensor,
                                    model: SentenceTransformer,
                                    n_resources_to_return: int = 5,
                                    print_verbose: bool = False):
        """
        Embeds a query with model and returns top k scores and indices from embeddings.
        """

        # Embed the query
        query_embedding = model.encode(query,
                                       convert_to_tensor=True)

        # Get dot product scores on embeddings
        start_time = timer()
        dot_scores = util.dot_score(query_embedding, embeddings)[0]
        end_time = timer()

        if print_verbose:
            print(
                f"[INFO]Time taken to get scores on {len(embeddings)} embeddings: {end_time - start_time:.5f} seconds.")

        scores, indices = torch.topk(input=dot_scores,
                                     k=n_resources_to_return)

        if print_verbose:
            print(f"[INFO]Top {n_resources_to_return} relevance return")
            print(f"[INFO]Scores:{scores}")
            print(f"[INFO]Indices:{indices}")

        return scores, indices

    @classmethod
    def print_wrapped(cls, text, wrap_length=80):
        wrapped_text = textwrap.fill(text, wrap_length)
        print(wrapped_text)

    @classmethod
    def print_top_results_and_scores(cls,
                                     query: str,
                                     embeddings: torch.tensor,
                                     model: SentenceTransformer,
                                     pages_and_chunks: list[dict],
                                     n_resources_to_return: int = 5):
        """
        Takes a query, retrieves most relevant resources and prints them out in descending order.

        Note: Requires pages_and_chunks to be formatted in a specific way (see above for reference).
        """

        scores, indices = cls.retrieve_relevant_resources(query=query,
                                                          embeddings=embeddings,
                                                          model=model,
                                                          n_resources_to_return=n_resources_to_return)

        print(f"Query: {query}\n")
        print("Results:")
        # Loop through zipped together scores and indicies
        for score, index in zip(scores, indices):
            print(f"Score: {score:.4f}")
            # Print relevant sentence chunk (since the scores are in descending order, the most relevant chunk will
            # be first)
            cls.print_wrapped(pages_and_chunks[index]["sentence_chunk"])
            # Print the page number too so we can reference the textbook further and check the results
            print(f"Page number: {pages_and_chunks[index]['page_number']}")
            print("\n")
