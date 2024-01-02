from CodeLanguageDetector.data.data_functions import get_processed_data
from CodeLanguageDetector.other_utils.files_io_utils import read_yaml
from loguru import logger
import copy
import os


DATA_CONFIG_PATH="../../../configs/data_config.yaml"
LOG_PATH = "../../../logs/running_logs.log"


if __name__=="__main__":
    logger.add(LOG_PATH, level="INFO")
    data_config = read_yaml(DATA_CONFIG_PATH)
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    processed_data_config = copy.deepcopy(data_config["processed_data"])
    
    for config in processed_data_config["get_processed_data"]:
        get_processed_data(**config)
    