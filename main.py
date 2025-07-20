"""Main entry point for the biography writer application.

This script initializes the processing pipeline and executes the biography generation steps.
It's importatnt to remark that this script takes the audio files from a directory, transcribes, and processes them to generate a biography.
It is recommended to perform a step-wise approach when having several audio files, and group them by life events or periods to avoid overwhelming the model with too much information at once.
This means repeating this execution for each life event or period you want to process.
If you want to use the biography writer, you need to have the audio files in the `input_directory` specified below.
"""

import argparse
from bwriter.utils import verify_path
from utils.logger import logger
from pathlib import Path
from bwriter.pipelines import ProcessingPipeline
from bwriter.transcriber import Transcriber


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Biography Writer Application")
    parser.add_argument(
        "-i", "--input_directory",
        type=str,
        required=True,
        help="Path to the directory containing audio files for processing."
    )
    parser.add_argument(
        "-tm", "--transcription_model",
        type=str,
        required=False,
        default=r"\models\whisper_large_v3",
    )
    parser.add_argument(
        "-st", "--skip_transcription",
        type=bool,
        required=False,
        default=False,
    )
    return parser.parse_args()


def create_biography():
    """Main function to create a biography."""
    args = parse_arguments()
    input_directory = Path(args.input_directory).resolve()
    model_path = Path(args.transcription_model).resolve()

    # Make sure that the input directory exists and is a directory
    if not input_directory.exists() or not input_directory.is_dir():
        raise RuntimeError(
            f"Error: Input directory '{input_directory}' does not exist or is not a directory."
        )

    # Transcribe audio files
    basepath = input_directory.parent
    transcription_output = basepath / "transcriptions"
    if not args.skip_transcription:
        verify_path(transcription_output)
        logger.info("Transcribing audio files...")
        transcriber = Transcriber(
            input_directory,
            transcription_output,
            model_path
        )
        transcriber.transcribe()
        logger.info("Transcription completed.")

    # Process transcriptions to generate biography
    logger.info("Processing transcriptions to generate biography...")
    # Define path for intermediate outputs
    corrected_path = basepath / "corrected_transcriptions"
    verify_path(corrected_path)
    merged_path = basepath / "merged_transcriptions"
    verify_path(merged_path)
    pipeline = ProcessingPipeline(
        input_directory=transcription_output,
        corrected_directory=corrected_path,
        merged_directory=merged_path,
    )
    # Run the processing pipeline
    logger.info("Running processing pipeline...")
    merge_kwargs = {"save_fig_path": merged_path/"clusters.png"}
    process_kwargs = {
        "max_length": 512,
        "chunk_overlap": 0
    }

    pipeline.run(
        process_kwargs=process_kwargs,
        save_files=True,
        merge_kwargs=merge_kwargs
    )
    logger.info("Processing completed. Biography generation finished.")


if __name__ == "__main__":
    create_biography()
    logger.info("Biography Writer Application executed successfully.")
