"""Useful methods to be used in the other classes."""

from pathlib import Path
from typing import List, Union

from utils.logger import Logger

logger = Logger.get_logger()


def read_file(filepath: Path, clean: bool = True) -> str:
    """Read the content of a file. If clean is True, it will convert the content to lowercase and strip whitespace."""
    with open(filepath, "r", encoding="latin-1") as file:
        file_content = file.read()
    if clean:
        file_content = file_content.lower().strip()
    logger.info(f"Reading file: {filepath}")

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


def verify_path(path: Path) -> None:
    """Verify if a path exists and is a directory.

    Args:
        path (Path): The path to verify.

    Raises:
        RuntimeError: If the path does not exist or is not a directory.
    """
    if not path.exists():
        logger.info(f"Error: Path '{path}' does not exist.")
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {path}")
    logger.info(f"Verified path: {path} exists.")
