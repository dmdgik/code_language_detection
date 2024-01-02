from typing import Any
from loguru import logger

import torch
from torch import Tensor
from transformers import AutoModel



def load_pretrained_hf_model(model_path: str) -> Any:
    """function for loading pretrained models from huggingface or local path
    
    Args:
        model_path (str): model name in huggingface or local path

    Raises:
        e: Exception

    Returns:
        Any: pretrained model
    """
    try:
        model = AutoModel.from_pretrained(model_path)
        logger.info(f"Model {model_path} loaded successfully")
    except Exception as e:
        logger.error(f"Model {model_path} loading failed. Exception: {e}")
        raise e
    return model


def save_pretrained_hf_model(model: Any, path_to_save: str) -> None:
    """function for saving pretrained huggingface model

    Args:
        model (Any): pretrained model
        path_to_save (str): model saving path

    Raises:
        e: Exception
    """
    try:
        model.save_pretrained(path_to_save)
        logger.info(f"Model saved to {path_to_save} successfully")
    except Exception as e:
        logger.error(f"Model saving to {path_to_save} failed. Exception: {e}")
        raise e
    

def load_and_save_hf_model(model_path: str, path_to_save: str) -> None:
    """Function for loading huggingface model and saving it locally 

    Args:
        model_path (str): pretrained model name or path
        path_to_save (str): model saving path
    """
    model = load_pretrained_hf_model(model_path)
    save_pretrained_hf_model(model, path_to_save)
    
    
def accuracy_score(outputs: Tensor, targets: Tensor) -> float:
    """accuracy score metric for tensors

    Args:
        outputs (Tensor): model outputs tensor
        targets (Tensor): fact target tensor

    Returns:
        float: accuracy score
    """
    return torch.sum(torch.argmax(outputs, dim=1) == targets).item() / len(outputs)


def load_model_torchscript(file_path: str, device: str='cpu') -> Any:
    """function for loading scripted model

    Args:
        file_path (str): torchscripted model path
        device (str, optional): model device. Defaults to 'cpu'.

    Raises:
        e: Exception

    Returns:
        Any: model
    """
    try: 
        model = torch.jit.load(file_path)
        model.eval()
        model.to(torch.device(device))
        logger.info(f"Model {file_path} loaded successfully")
    except Exception as e:
        logger.error(f"Model {file_path} loading failed. Exception: {e}")
        raise e
    return model
