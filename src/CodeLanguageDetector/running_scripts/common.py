from CodeLanguageDetector.other_utils.files_io_utils import read_yaml
from CodeLanguageDetector.data.utils import load_and_save_tokenizer
from CodeLanguageDetector.models.utils import load_and_save_hf_model
from loguru import logger
import copy


COMMON_CONFIG_PATH = "../../../configs/common.yaml"
LOG_PATH = "../../../logs/running_logs.log"


if __name__=="__main__":
    logger.add(LOG_PATH, level="INFO")
    common_config = read_yaml(COMMON_CONFIG_PATH)
    
    tokenizer_config = copy.deepcopy(common_config["tokenizer"])
    base_model_config = copy.deepcopy(common_config["base_model"])

    load_and_save_tokenizer(**tokenizer_config)
    load_and_save_hf_model(**base_model_config)