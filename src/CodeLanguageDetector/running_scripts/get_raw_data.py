from CodeLanguageDetector.data.data_functions import get_language_dataset
from CodeLanguageDetector.other_utils.files_io_utils import read_yaml
from loguru import logger
import copy


DATA_CONFIG_PATH="../../../configs/data_config.yaml"
SECRETS_CONFIG_PATH="../../../configs/secrets.yaml"
LOG_PATH = "../../../logs/running_logs.log"


if __name__=="__main__":
    logger.add(LOG_PATH, level="INFO")
    data_config = read_yaml(DATA_CONFIG_PATH)
    secrets_config = read_yaml(SECRETS_CONFIG_PATH)
    
    raw_data_config = copy.deepcopy(data_config["raw_data"])
    raw_data_config["get_language_dataset"]["auth_token"] = secrets_config["HUGGINGFACE_AUTH_TOKEN"]
    
    get_language_dataset(**raw_data_config["get_language_dataset"])
    
    