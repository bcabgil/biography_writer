"""Classes for processing the transcribed texts."""

import json
import os
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import openai
import pandas as pd
import umap
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.base import RunnableSequence
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from bwriter.utils import read_file, verify_path, write_file
from utils.logger import logger


class Processor:
    """Use a selected OpenAI llm to process texts."""

    def __init__(self, llm_name: str = "gpt-4o-mini"):
        """Initialize the processor with a given LLM name."""
        os.environ["OPENAI_API_KEY"] = read_file(
            os.getenv("OPENAI_API_KEY_PATH", default=r"openai_key.txt"), clean=False
        )
        self.llm = ChatOpenAI(model=llm_name)

    def get_recap(self, inputs: Union[Dict, List]):
        """Get a summary of the text using the LLM."""
        # Retrieve the main points of the text
        prompt = PromptTemplate.from_template(self.prompts["recap"])
        summary_chain = prompt | self.llm | StrOutputParser()
        if isinstance(inputs, list):
            return summary_chain.batch(inputs)

        return summary_chain.invoke(inputs)

    @staticmethod
    def create_document_library(input_path: Path) -> List[Document]:
        """Read the documents from a directory and create a library of docs."""
        # Create a library of documents
        file_library = []

        logger.info("Creating document library.")
        for i, file in enumerate(tqdm(input_path.iterdir())):
            if file.is_dir():
                continue
            # Read the content of the file
            file_content = read_file(file)
            # Create a Document object for each file
            file_library.append(
                Document(
                    page_content=file_content,
                    metadata={"source": str(file)},
                    id=i,
                )
            )
        return file_library

    @staticmethod
    def split_text_into_chunks(
        file_library: List[Document], max_length: int = 1000, chunk_overlap: int = 100
    ) -> List[Document]:
        """Read the documents and divide them into smaller chunks.

        Args:
            file_library (List[Document]): List of documents to be split.
            max_length (int): Maximum length of each chunk. Defaults to 1000.
            chunk_overlap (int): Overlap between chunks. Defaults to 100.
        """
        separators = [
            "\n\n",
            "\n",
            ".",
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",
        ]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_length,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            separators=separators,
        )
        document_splits = text_splitter.split_documents(file_library)
        return document_splits

    @staticmethod
    def save_text(
        document_splits: List[Document], output_path: Path, merged_texts: bool = False
    ) -> None:
        """Save the input text to the stated location.

        Args:
            document_splits (List[Document]): List of document splits to be saved.
            output_path (Path): Path where the texts will be saved.
            merged_texts (bool): Whether the input texts are the result of the merging step.
            Defaults to False.
        """
        verify_path(output_path)

        logger.info(f"Saving texts in {output_path}")
        if merged_texts:
            for idx, text in document_splits.items():
                # Save text
                filename = output_path / f"cluster_{idx}.txt"
                write_file(filename, text)

        # Otherwise, save each split as a separate file
        else:
            for split in document_splits:
                source = Path(split["source"])
                counter = split["counter"]
                text = split["text"]
                output_file = output_path / f"{source.stem}_{counter}.txt"
                write_file(filepath=output_file, text=text)

    @staticmethod
    def read_text_files(input_path: Path) -> List[str]:
        """Read text text from a given path."""
        texts = []
        logger.info(f"Reading text files from {input_path}")
        for file in input_path.iterdir():
            if file.is_dir():
                continue
            with open(file, "r") as file_i:
                file_content = file_i.read()
            texts.append({"text": file_content, "source": file})
        return texts

    @staticmethod
    def get_embedding(
        text: List[str], model: str = "text-embedding-3-small"
    ) -> List[np.array]:
        """
        Get the embedding for a given text using OpenAI's embedding API.

        Return: List of embeddings for each word in the text.
        """
        logger.info(f"Getting embeddings for the text with model {model}.")
        response = openai.embeddings.create(input=text, model=model)
        return [np.array(emb.embedding) for emb in response.data]

    @staticmethod
    def reduce_embeddings(
        embeddings: List[np.array], method: str, kwargs: Dict = {}
    ) -> List[int]:
        """
        Reduce the dimensionality of the embeddings using a given method.

        Args:
            embeddings (List[np.array]): List of embeddings to reduce.
            method (str): Method to use for dimensionality reduction (e.g., "umap").
            kwargs (Dict): Additional arguments for the dimensionality reduction method.

        Return: List of cluster labels for each word in the text.
        """
        if method == "umap":
            # Use UMAP to cluster the embeddings
            if kwargs == {}:
                kwargs = {"n_neighbors": 3, "min_dist": 0.3, "metric": "cosine"}
            logger.info("Reducing the dimensionality using UMAP.")
            reducer = umap.UMAP(**kwargs)
            embedding_umap = reducer.fit_transform(embeddings)
            return embedding_umap
        else:
            raise ValueError(
                "Invalid dimensionality reduction method. The only supported method is 'umap'."
            )

    @staticmethod
    def cluster_embeddings(
        embeddings: List[np.array], method: str = "dbscan", cluster_args: Dict = {}
    ) -> List[int]:
        """
        Get the clusters for a list of vectors using a given method.

        Args:
            embeddings (List[np.array]): List of embeddings to cluster.
            method (str): Clustering method to use (e.g., "dbscan").
            cluster_args (Dict): Additional arguments for the clustering method.

        Return: List of cluster labels for each item in the input list.
        """
        if method == "dbscan":
            # Use DBSCAN to cluster the embeddings
            if cluster_args == {}:
                cluster_args = {"eps": 0.9, "min_samples": 3, "metric": "euclidean"}
            logger.info("Clustering the embeddings using DBSCAN.")
            clustering = DBSCAN(**cluster_args)
            return clustering.fit_predict(embeddings)
        else:
            raise ValueError(
                "Invalid clustering method. The only supported method is 'dbscan'."
            )

    @staticmethod
    def get_clusters_from_text(
        texts: List[str],
        embedding_model: str = "text-embedding-3-small",
        dim_red_method: str = "umap",
        cluster_method: str = "dbscan",
        dim_red_args: Dict = {},
        cluster_args: Dict = {},
        save_fig_path: str = None,
    ) -> List[int]:
        """
        Get the clusters for a given list of texts using a dimensionality
        reduction and clustering method.

        Args:
            texts (List[str]): List of texts to cluster.
            embedding_model (str): Model to use for embeddings.
            Defaults to "text-embedding-3-small".
            dim_red_method (str): Method to use for dimensionality reduction.
            Defaults to "umap".
            cluster_method (str): Clustering method to use.
            Defaults to "dbscan".
            dim_red_args (Dict): Additional arguments for the dime reduction method.
            cluster_args (Dict): Additional arguments for the clustering method.
            save_fig_path (str, optional): Path to save the clustering figure.
            Defaults to None.

        Return: List of cluster labels for each word in the text.
        """
        # Get the embeddings for the text
        embeddings = Processor.get_embedding(texts, embedding_model)

        # Reduce the dimensionality of the embeddings
        reduced_embeddings = Processor.reduce_embeddings(
            embeddings, dim_red_method, dim_red_args
        )

        # Cluster the embeddings
        clusters = Processor.cluster_embeddings(
            reduced_embeddings, cluster_method, cluster_args
        )

        # Save the figure if a path is given
        if save_fig_path:
            plt.figure(figsize=(10, 10))
            plt.scatter(
                reduced_embeddings[:, 0],
                reduced_embeddings[:, 1],
                c=clusters,
            )
            plt.gca().set_aspect("equal", "datalim")
            plt.title("Clusters", fontsize=24)
            plt.colorbar()
            plt.savefig(save_fig_path)
            logger.info(f"Figure saved in {save_fig_path}")

        return clusters


class Corrector(Processor):
    """Correct texts with a selected OpenAI llm."""
    def __init__(self, llm_name: str = "gpt-4o-mini"):
        super().__init__(llm_name)
        self.prompts = json.load(open("bwriter/prompts/processor_prompts.json", "r"))[
            "processor_prompts"
        ]["correction"]

    def correction_chain(self) -> RunnableSequence:
        """Correct a text transcript.

        Returns:
            RunnableSequence: A chain that takes a text and corrects it.
        """
        prompt = PromptTemplate.from_template(self.prompts)
        # Create a chain that takes a text and returns the corrected text
        literary_chain = prompt | self.llm | StrOutputParser()

        return literary_chain

    def get_text_corrections(self, document_splits: List[Document]) -> Dict:
        """Correct each document split.

        Args:
            document_splits (List[Document]): List of document splits to be corrected.
        """
        correction_chain = self.correction_chain()

        splits_corretgits = []
        prev_source = ""
        # Counter to keep track of the number of splits for each source
        counter = 0
        logger.info("Correcting text using correction chain")
        for ds in tqdm(document_splits, total=len(document_splits)):
            source = ds.metadata["source"]
            # If the source is different from the previous one, reset the counter
            # Otherwise, increment the counter
            if source != prev_source:
                prev_source = source
                counter = 0
            else:
                counter += 1

            text = ds.page_content
            splits_corretgits.append(
                {
                    "source": source,
                    "counter": counter,
                    "text": correction_chain.invoke({"text": text}),
                }
            )

        return splits_corretgits


class Sorter(Processor):
    """Sorts texts with a selected OpenAI llm."""
    def __init__(self, llm_name: str = "gpt-4o-mini"):
        super().__init__(llm_name)
        self.prompts = json.load(open("bwriter/prompts/processor_prompts.json", "r"))[
            "processor_prompts"
        ]["order"]

    def llm_order_chain(self, text_to_order: List[str]) -> str:
        """Create chain to order the texts"""
        prompt_merge = PromptTemplate.from_template(self.prompts)
        order_chain = prompt_merge | self.llm | StrOutputParser()

        return order_chain.invoke({"text": text_to_order})

    def sort_texts(self, input_texts: Dict[int, str]) -> str:
        """Order the texts according to the clusters they belong to."""
        # Convert to single string
        numbered_clusters = [
            f"{cluster}. {text}" for cluster, text in input_texts.items()
        ]
        # Order the texts
        ordered_sequence = self.llm_order_chain("\n\n".join(numbered_clusters))
        ordered_list = eval(ordered_sequence)
        # Join text in a single string
        try:
            ordered_text = "\n\n".join(
                [input_texts[cluster] for cluster in ordered_list]
            )
        except RuntimeError as e:
            logger.error(f"Error ordering texts: {e}")
            raise RuntimeError(
                f"""The ordering of the texts failed.
                This may be caused by the ordering returned by the llm.
                llm clusters: {ordered_list}
                input clusters: {list(input_texts.keys())}
                Please check the input texts and the llm ordering.
                """
            )

        return ordered_text


class Merger(Processor):
    """Merges texts with a selected OpenAI llm."""
    def __init__(self, llm_name: str = "gpt-4o-mini"):
        super().__init__(llm_name)
        self.prompts = json.load(open("bwriter/prompts/processor_prompts.json", "r"))[
            "processor_prompts"
        ]["merge"]
    
    def llm_merge_texts(self, text_to_merge: List[str]) -> str:
        """Merge a list of texts using the LLM.

        Args:
            text_to_merge (List[str]): List of texts to merge.
        """
        prompt_merge = PromptTemplate.from_template(self.prompts)
        merge_chain = prompt_merge | self.llm | StrOutputParser()

        return merge_chain.invoke({"text": "/n/n".join(text_to_merge)})

    def merge_texts(self, texts: List[str], clusters: List[str]) -> Dict[str, str]:
        """Merge a list of texts according to the clusters they belong to."""
        # Create a dataframe of texts and clusters
        df = pd.DataFrame({"text": texts, "cluster": clusters})
        unique_clusters = df["cluster"].unique()

        text_per_cluster = {}
        # Merge the texts in each cluster
        logger.info("Merging texts in each cluster.")
        for cluster in tqdm(unique_clusters, total=len(unique_clusters)):
            # Get the texts in the cluster
            texts_to_merge = df[df["cluster"] == cluster]["text"].tolist()
            # Merge the texts in the cluster
            merged_cluster = self.llm_merge_texts(texts_to_merge)
            text_per_cluster[cluster] = merged_cluster

        return text_per_cluster
