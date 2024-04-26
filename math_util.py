import torch


class MathUtil:
    def __init__(self):
        pass

    @classmethod
    def dot_product(cls, vector1, vector2):
        return torch.dot(vector1, vector2)

    @classmethod
    def cosine_similarity(cls, vector1, vector2):
        dot_prod = torch.dot(vector1, vector2)

        norm_vector1 = torch.sqrt(torch.sum(vector1**2))
        norm_vector2 = torch.sqrt(torch.sum(vector2**2))

        return dot_prod / (norm_vector1 * norm_vector2)
