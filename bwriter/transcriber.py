"""This class is used to transcribe the audios sent."""
from pathlib import Path
from transformers import pipeline
from tqdm import tqdm
import torch
import os
from utils.logger import logger
from typing import Union


class Transcriber:
    """Class to transcribe audios."""

    def __init__(self, audio_path: Path, transcriptions_path: Path, model_path: Path):
        """Initialize the Transcriber class.

        Args:
            audio_path (Path): Directory containing the audio files to transcribe.
            transcriptions_path (Path): Directory where transcriptions will be saved.
            model_path (Path): Path to the pre-trained transcription model, whisper from OpenAI recommended.
        """
        self.audio_path = audio_path
        self.transcriptions_path = transcriptions_path
        self.model_path = model_path

    @staticmethod
    def save_transcription(text: str, transcription_path: Path) -> None:
        """Save the transcription to a file."""
        with open(transcription_path, "w") as f:
            f.write(text)
        logger.info(f"Transcription saved in : {transcription_path}")

    @staticmethod
    def get_model(model_path: Path) -> pipeline:
        """Get the transcription model"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Model set to run in: {device}")
        transcriber = pipeline(
            "automatic-speech-recognition", model=model_path, device=device
        )
        logger.info(f"Model saved from: {model_path}")
        return transcriber

    def get_new_filename_and_transcription_filename(
        self, filename: Path
    ) -> Union[Path, Path]:
        """Rename original filename with spaces and get transcription filename."""
        # Replace spaces by _
        new_basename = str(filename.stem).replace(" ", "_")
        # Renamed file
        new_audio_filepath = self.audio_path / (new_basename + ".wav")
        destination = self.transcriptions_path / (new_basename + ".txt")

        return new_audio_filepath, destination

    def transcribe(self):
        """Transcribe all files from the audios directory."""
        # Get model
        transcriber = self.get_model(self.model_path)

        for file in tqdm(self.audio_path.iterdir()):
            if file.is_dir():
                continue
            logger.info(f"Transcribing file: {file}")
            # Rename file if it contains spaces
            (
                new_audio_filepath,
                destination,
            ) = self.get_new_filename_and_transcription_filename(file)

            # If the file has already been transcribed continue
            if destination in list(self.transcriptions_path.iterdir()):
                logger.info(f"{file} already transcribed. Moving on to the next audio.")
                continue

            # Rename audio
            os.rename(str(file), new_audio_filepath)
            logger.info(f"Audio file renamed from {file} -> {new_audio_filepath}")

            # Start transcribing
            logger.info("Transcribing....")
            result = transcriber(
                str(new_audio_filepath),
                generate_kwargs={"language": "catalan"}
            )

            # Save transcription
            Transcriber.save_transcription(result["text"], destination)
