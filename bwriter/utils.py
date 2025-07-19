"""Useful methods to be used in the other classes."""

from pathlib import Path
from typing import List, Union
from utils.logger import Logger
logger = Logger.get_logger()


def read_file(filepath: Path) -> str:
    """Read the content of a file."""
    with open(filepath, "r") as file:
        file_content = file.read()

    return file_content


def write_file(filepath: Path, text: Union[str, List[str]]) -> None:
    """Write text to a file.

    Args:
        filepath (Path): Path to the file where the text will be saved.
        text (Union[str, List[str]]): Text or list of strings to write to the file.
    """
    logger.info(f"Writing text to file: {filepath}")
    with open(filepath, "w") as save_file:
        # Writing data to a file
        if isinstance(text, list):
            save_file.writelines(text)
        else:
            save_file.write(text)
