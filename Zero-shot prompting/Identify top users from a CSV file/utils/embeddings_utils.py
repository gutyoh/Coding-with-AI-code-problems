import openai
from typing import List
from tenacity import retry, stop_after_attempt, wait_random_exponential
from scipy import spatial


class EmbeddingsUtils:
    _api_key = None  # Class variable to store the API key

    @classmethod
    def set_openai_api_key(cls, key: str):
        cls._api_key = key
        openai.api_key = cls._api_key

    @classmethod
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def get_embedding(cls, text: str, model="text-embedding-ada-002", **kwargs) -> List[float]:
        """
        Get embedding for a given text using the specified OpenAI model.
        """
        if cls._api_key is None:
            raise ValueError("OpenAI API key is not set.")

        text = text.replace("\n", " ")
        response = openai.embeddings.create(input=[text], model=model, **kwargs)
        return response.data[0].embedding

    @staticmethod
    def distances_from_embeddings(query_embedding: List[float], embeddings: List[List[float]],
                                  distance_metric="cosine") -> List[float]:
        """
        Return the distances between a query embedding and a list of embeddings.
        """
        distance_metrics = {
            "cosine": spatial.distance.cosine,
            "L1": spatial.distance.cityblock,
            "L2": spatial.distance.euclidean,
            "Linf": spatial.distance.chebyshev,
        }
        return [distance_metrics[distance_metric](query_embedding, embedding) for embedding in embeddings]
