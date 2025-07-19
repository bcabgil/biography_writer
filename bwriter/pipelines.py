"""Pipeline to generate the biography."""
from pathlib import Path
from typing import Dict, List, Union

from bwriter.processing import Processor
from bwriter.utils import write_file
from utils.logger import logger


class ProcessingPipeline:
    """
    Processing pipeline to generate the biography.

    This pipeline reads text files and processes them in the following steps:
    1. Splits the text into smaller chunks.
    2. Corrects the text chunks using a language model.
    3. Merges similar text chunks into clusters.
    4. Orders the clusters based on their content.
    5. Saves the ordered clusters to a text file.
    """

    def __init__(
        self,
        input_directory: Path,
        corrected_directory: Path,
        merged_directory: Path,
        llm_correction: str = "gpt-4o-mini",
        llm_merge: str = "gpt-4o",
        llm_sorter: str = "gpt-4o-mini",
    ):
        """Initialize the processing pipeline.

        Args:
            input_directory (Path): Directory containing the input text files.
            corrected_directory (Path): Directory to save corrected text files.
            merged_directory (Path): Directory to save merged text files.
            llm_correction (str, optional): OpenAI LLM used for correction.
            Defaults to "gpt-4o-mini".
            llm_merge (str, optional): OpenAI LLM used in the merge step.
            Defaults to "gpt-4o".
            llm_sorter (str, optional): OpenAI LLM used for sorting.
            Defaults to "gpt-4o-mini".
        """
        self.input_directory = input_directory
        self.corrected_directory = corrected_directory
        self.merged_directory = merged_directory
        self.corrector = Processor(llm_name=llm_correction)
        self.merger = Processor(llm_name=llm_merge)
        self.sorter = Processor(llm_name=llm_sorter)

    def process_files(
        self, save_files: bool = True, **kwargs
    ) -> Dict[str, Union[str, List[str]]]:
        """Split and correct inptut files

        Args:
            save_files (bool, optional): Whether to save the corrected files. 
            Defaults to True.
            **kwargs: Additional keyword arguments for text processing.
        """
        # Create document library
        documents = self.corrector.create_document_library(
            self.input_directory
        )
        # Split documents
        split_documents = self.corrector.split_text_into_chunks(
            documents,
            **kwargs
        )
        # Correct documents
        corrected_splits = self.corrector.get_text_corrections(split_documents)
        if save_files:
            # Save corrected splits
            self.corrector.save_text(
                corrected_splits,
                self.corrected_directory
            )
        return corrected_splits

    def merge_similar_splits(
        self,
        text_splits: List[str],
        save_files: bool = True,
        **kwargs
    ) -> Dict[int, str]:
        """Merge text chuncks with similar content

        Args:
            text_splits (List[str]): List of text chunks to merge.
            save_files (bool, optional): Whether to save the merged files.
            Defaults to True.
            **kwargs: Additional keyword arguments for merging.
        """
        # Get text clusters
        text_clusters = self.merger.get_clusters_from_text(
            text_splits,
            **kwargs
        )
        # Merge clusters
        merged_clusters = self.merger.merge_texts(text_splits, text_clusters)
        # Save merged clusters
        if save_files:
            self.merger.save_text(merged_clusters, self.merged_directory)
        return merged_clusters

    def order_clusters(self, text_clusters: Dict[int, str]) -> List[str]:
        """Order clusters based on their content.

        Args:
            text_clusters (Dict[int, str]): Dictionary of text clusters.
        """
        ordered_text = self.sorter.sort_texts(text_clusters)
        # Save ordered clusters in a text file
        text_filepath = self.merged_directory / "ordered_clusters.txt"
        write_file(text_filepath, ordered_text)
        return ordered_text

    def run(
            self,
            process_kwargs,
            merge_kwargs,
            save_files: bool = True
    ) -> str:
        """Run the pipeline

        Args:
            process_kwargs (dict): Keyword arguments for text processing.
            merge_kwargs (dict): Keyword arguments for the merging step.
            save_files (bool, optional): Whether to save the processed files.
            Defaults to True.
        """
        corrected_splits = self.process_files(save_files, **process_kwargs)
        texts_to_merge = [text["text"] for text in corrected_splits]
        merged_clusters = self.merge_similar_splits(
            texts_to_merge, save_files, **merge_kwargs
        )
        ordered_text = self.order_clusters(merged_clusters)
        logger.info("Pipeline run completed.")
        return ordered_text
