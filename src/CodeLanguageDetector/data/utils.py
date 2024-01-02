from typing import Any
from loguru import logger
from transformers import AutoTokenizer


def load_tokenizer(tokenizer_path: str) -> Any:
    """Tokenizer loading function

    Args:
        tokenizer_name (str): pretrained tokenizer name or path

    Raises:
        e: Exception

    Returns:
        Any: tokenizer
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logger.info(f"Tokenizer {tokenizer_path} loaded successfully")
    except Exception as e:
        logger.error(f"Tokenizer {tokenizer_path} loading failed. Exception: {e}")
        raise e
    return tokenizer


def save_tokenizer(tokenizer: Any, path_to_save: str) -> None:
    """Tokenizer saving function

    Args:
        tokenizer (Any): tokenizer
        path_to_save (str): tokenizer saving path

    Raises:
        e: Exception
    """
    try:
        tokenizer.save_pretrained(path_to_save)
        logger.info(f"Tokenizer saved to {path_to_save} successfully")
    except Exception as e:
        logger.error(f"Tokenizer saving to {path_to_save} failed. Exception: {e}")
        raise e


def load_and_save_tokenizer(tokenizer_path: str, path_to_save: str) -> None:
    """Function for loading tokenizer and saving it locally 

    Args:
        tokenizer_name (str): pretrained tokenizer name or path
        path_to_save (str): tokenizer saving path
    """
    tokenizer = load_tokenizer(tokenizer_path)
    save_tokenizer(tokenizer, path_to_save)
