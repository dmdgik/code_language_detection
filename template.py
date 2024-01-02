if __name__ == "__main__":
    import os
    from pathlib import Path
    import logging
    import yaml

project_name = "CodeLanguageDetector"

if __name__ == "__main__":
    
    list_of_folders = [
        "./data/raw/",
        "./data/processed/",
        "./logs/",
        "./models/obtained/",
        "./models/pretrained/",
        f"./src/{project_name}/running_scripts/",
    ]

    list_of_files = [
        "./configs/secrets.yaml",
        "./configs/common.yaml",
        "./configs/data_config.yaml",
        f"./src/{project_name}/__init__.py",
        f"./src/{project_name}/data/__init__.py",
        f"./src/{project_name}/models/__init__.py",
        f"./src/{project_name}/other_utils/__init__.py",
        "./notebooks/drafts.ipynb",
        "./requirements.txt",
        "./Makefile",
        "./setup.py",
    ]

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s:")

    for filepath in list_of_files:
        
        filepath = Path(filepath)
        filedir, filename = os.path.split(filepath)
        
        if (filedir != "") and (not os.path.exists(filedir)):
            os.makedirs(filedir, exist_ok=True)
            logging.info(f"Creating directory: {filedir}")
        
        if not os.path.exists(filepath):
            with open(filepath, "w") as f:
                logging.info(f"Creating empty file: {filepath}")
        else:
            logging.info(f"{filepath} is already exists")


    for dirpath in list_of_folders:
        
        dirpath = Path(dirpath)
        os.makedirs(dirpath, exist_ok=True)
        logging.info(f"Creating directory: {dirpath}")