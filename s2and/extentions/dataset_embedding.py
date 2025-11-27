import asyncio
import json
import os
from abc import ABC, abstractmethod
from os.path import join
from typing import Dict, List, Optional, Tuple

import numpy as np
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm.asyncio import tqdm as atqdm

from s2and import logger


class EncoderModel(ABC):
    """Base class for all encoder models"""

    @abstractmethod
    async def forward(self, title: str, abstract: str) -> List[float]:
        """
        Implement forward pass for the model.
        :param title: Input title to encode
        :param abstract: Input abstract to encode
        :return: Embedding vector as a list
        """
        pass


class GeminiModel(EncoderModel):
    """Class that implements model using Google Gemini API"""

    def __init__(
        self,
        gcp_project: str,
        gcp_location: str,
        model: str,
        task_type: Optional[str] = None,
        truncate_dim: Optional[int] = None,
    ) -> None:
        """
        Initialize Gemini model
        :param gcp_project: GCP project ID
        :param gcp_location: GCP location
        :param model: The name of the Gemini embedding model
        :param task_type: Task type for embedding (e.g., 'RETRIEVAL_DOCUMENT', 'RETRIEVAL_QUERY')
        :param truncate_dim: If set, truncates the embeddings to this dimension
        """
        self.gcp_project: str = gcp_project
        self.gcp_location: str = gcp_location
        self.model: str = model
        self.task_type: Optional[str] = task_type
        self.truncate_dim: Optional[int] = truncate_dim
        self.client: genai.Client = genai.Client(
            vertexai=True, project=gcp_project, location=gcp_location
        )

    async def forward(self, title: str, abstract: str) -> List[float]:
        """
        Implement model's forward method
        :param text: Input text to encode
        :return: Embedding vector as a list
        """
        text = title + ". " + abstract
        return await self._encode_with_retry(text)

    @retry(wait=wait_fixed(1), stop=stop_after_attempt(10))
    async def _encode_with_retry(self, text: str) -> List[float]:
        """
        Encode text with retry logic
        :param text: Input text to encode
        :return: Embedding vector as a list
        """
        config = types.EmbedContentConfig(
            task_type=self.task_type, output_dimensionality=self.truncate_dim
        )

        try:
            response = await self.client.aio.models.embed_content(
                model=self.model, contents=[text], config=config
            )
            embedding = response.embeddings[0].values
            # Normalize embedding to unit length
            if self.truncate_dim is not None:
                embedding: np.ndarray = np.array(embedding) / np.linalg.norm(embedding)
                return embedding.tolist()
            else:
                return embedding
        except genai.errors.ClientError as e:
            logger.error(f"Error encoding text: {e}")
            # Return zero vector on error
            dim = self.truncate_dim if self.truncate_dim is not None else 3072
            return [0.0] * dim


async def embed_s2and(
    model: EncoderModel,
    model_name: str,
    data_dir: str,
    embeddings_dir: str,
    dataset_names: List[str] = ["pubmed", "aminer", "zbmath", "kisti", "arnetminer"],
    batch_size: int = 10,
) -> None:
    """
    Embed datasets regarding s2and framework
    :param model: Model that implements forward method and infers vectors
    :param model_name: Name of the model
    :param data_dir: Directory of s2and dataset
    :param embeddings_dir: Directory embeddings will be saved
    :param dataset_names: List of dataset names to embed
    :param batch_size: Number of papers to process concurrently in each batch
    """
    embeddings_dir = join(embeddings_dir, model_name)

    for dataset_name in dataset_names:
        logger.info(f"Embedding S2AND {dataset_name}...")
        with open(join(data_dir, dataset_name, f"{dataset_name}_papers.json")) as f:
            papers = json.load(f)
        with open(join(data_dir, dataset_name, f"{dataset_name}_signatures.json")) as f:
            signature_to_paper_id = json.load(f)

        if not os.path.isdir(embeddings_dir):
            os.mkdir(embeddings_dir)

        embeddings: Dict[str, List[float]] = {}

        # Collect unique papers to process
        papers_to_process: List[Tuple[str, str, str]] = []
        for signature_id in signature_to_paper_id.keys():
            paper_id = signature_to_paper_id[signature_id]["paper_id"]
            paper_id = str(paper_id)
            if paper_id not in embeddings:
                title = papers[paper_id]["title"] or ""
                abstract = papers[paper_id]["abstract"] or ""
                papers_to_process.append((paper_id, title, abstract))
                # Mark as seen to avoid duplicates in batch
                embeddings[paper_id] = []

        # Process in batches
        for i in atqdm(
            range(0, len(papers_to_process), batch_size),
            desc=f"Processing batches for {dataset_name}",
        ):
            batch = papers_to_process[i : i + batch_size]  # noqa: E203

            # Gather results for the batch
            tasks = [model.forward(title, abstract) for _, title, abstract in batch]
            results = await asyncio.gather(*tasks)

            # Store results
            for (paper_id, _, _), embedding in zip(batch, results):
                embeddings[paper_id] = embedding

        with open(join(embeddings_dir, f"{dataset_name}_embeddings.json"), "w") as f:
            json.dump(embeddings, f)
