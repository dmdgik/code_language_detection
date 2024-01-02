from CodeLanguageDetector.other_utils.files_io_utils import read_yaml
from CodeLanguageDetector.models.models import CodeLanguageModelBERT
from pytorch_model_wrapper import FittableModelAvailables, FittableModelWrapper
from CodeLanguageDetector.models.datasets import CodeLanguageDataset
from CodeLanguageDetector.models.utils import accuracy_score
import torch
import argparse
import os
from loguru import logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="experiment config parser")
    parser.add_argument(
        "--experiment_config",
        type=str,
        help="provide your experiment config yaml file"
    )
    experiment_args = parser.parse_args()
    
    EXPERIMENT_CONFIG_FILE = experiment_args.experiment_config
    SECRETS_CONFIG_FILE = "../../../configs/secrets.yaml"
    experiment_config = read_yaml(EXPERIMENT_CONFIG_FILE)
    additional_fit = experiment_config["extra"]["additional_changes"][0]["change"]
    
    experiment_availables = {
        "CodeLanguageDataset" : CodeLanguageDataset,
        "CodeLanguageModelBERT" : CodeLanguageModelBERT,
        "accuracy_score" : accuracy_score,
        "CrossEntropyLoss" : torch.nn.CrossEntropyLoss,
        "ExponentialLR" : torch.optim.lr_scheduler.ExponentialLR,
    }
    
    try:
        secrets_config = read_yaml(SECRETS_CONFIG_FILE)
        os.environ["AWS_S3_ACCESS_KEY_ID"] = secrets_config["AWS_S3_ACCESS_KEY_ID"]
        os.environ["AWS_S3_SECRET_ACCESS_KEY"] = secrets_config["AWS_S3_SECRET_ACCESS_KEY"]
    except:
        pass
    
    FittableModelAvailables.metrics["accuracy"] = [experiment_availables["accuracy_score"], 1]
    FittableModelAvailables.schedulers["ExponentialLR"] = experiment_availables["ExponentialLR"]
    
    dataset_train = experiment_availables[experiment_config["extra"]["dataset"]["dataset_class"]](**experiment_config["extra"]["dataset"]["dataset_params"]["train"])
    dataset_valid = experiment_availables[experiment_config["extra"]["dataset"]["dataset_class"]](**experiment_config["extra"]["dataset"]["dataset_params"]["valid"])
    dataset_test = experiment_availables[experiment_config["extra"]["dataset"]["dataset_class"]](**experiment_config["extra"]["dataset"]["dataset_params"]["test"])
    model = experiment_availables[experiment_config["extra"]["model"]["model_class"]](**experiment_config["extra"]["model"]["model_params"])
    criterion = experiment_availables[experiment_config["extra"]["criterion"]]()
    
    model = FittableModelWrapper(model, experiment_config)
    model.fit(criterion, dataset_train, dataset_valid)
    model.evaluate(criterion, dataset_test)
    
    try:
        model.s3_upload()
    except:
        logger.log("INFO", "S3 uploading failed")
        
    model.fit(criterion, dataset_train, dataset_valid, additional_fit)
    model.evaluate(criterion, dataset_test)
    model.clean_snapshots()
    
    try:
        model.s3_upload()
    except:
        logger.log("INFO", "S3 uploading failed")
