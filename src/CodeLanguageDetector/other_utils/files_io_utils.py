import yaml
import pickle
from typing import Dict, Optional
from loguru import logger
import os


def read_yaml(path_to_yaml_file: str) -> Optional[Dict]:
    """function for reading yaml file

    Args:
        path_to_yaml_file (str): path to yaml file

    Raises:
        e: Exception

    Returns:
        Optional[Dict]: dict from yaml file
    """
    try:
        
        with open(path_to_yaml_file, "r") as f:
            result = yaml.safe_load(f)
        logger.info(f"Yaml file {path_to_yaml_file} read successfully")
    except Exception as e:
        logger.error(f"Yaml file {path_to_yaml_file} can't be read. Exception: {e}")
        raise e
    return result


def read_dict_pkl(path_to_pkl_file: str) -> Optional[Dict]:
    """function for reading dict pkl file

    Args:
        path_to_pkl_file (str): path to pkl file

    Raises:
        e: Exception

    Returns:
        Optional[Dict]: dict from pkl file
    """
    try:
        with open(path_to_pkl_file, "rb") as f:
            result = pickle.load(f)
        logger.info(f"Pkl file {path_to_pkl_file} read successfully")
    except Exception as e:
        logger.error(f"Pkl file {path_to_pkl_file} can't be read. Exception: {e}")
        raise e
    return result


def write_dict_pkl(dictionary: Dict, path_to_pkl_file: str, exist_ok: bool=False) -> None:
    """function for writing dict into pkl file

    Args:
        dictionary (Dict): dictionary
        path_to_pkl_file (str): path to pkl file
        exist_ok (bool, optional): overwriting file condition. 
            If False - file will not overwrite and dict will not write to file. 
            If True - file will overwrite with dictionary passed
            Defaults to False.

    Raises:
        e: Exception
    """
    if exist_ok:
        try:
            if os.path.exists(path_to_pkl_file):
                logger.warning(f"File {path_to_pkl_file} already exist. It will be overwrite with passed dictionary")
            with open(path_to_pkl_file, "wb") as f:
                pickle.dump(dictionary, f)
            logger.info(f"Dictionary passed to file {path_to_pkl_file}")
        except Exception as e:
            logger.error(f"Dictionary can't be passed to file {path_to_pkl_file} , Exception: {e}")
            raise e
    else:
        if not os.path.exists(path_to_pkl_file):
            try:
                with open(path_to_pkl_file, "wb") as f:
                    pickle.dump(dictionary, f)
                logger.info(f"Dictionary passed to file {path_to_pkl_file}")
            except Exception as e:
                logger.error(f"Dictionary can't be passed to file {path_to_pkl_file} , Exception: {e}")
                raise e
        else:
            logger.warning(f"File {path_to_pkl_file} already exist. Check file, please! Delete it or set exist_ok=True")
