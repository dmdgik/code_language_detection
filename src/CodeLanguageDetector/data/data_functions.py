import os
import gc
from tqdm import tqdm
from random import randint
from typing import Any
from loguru import logger

import sqlite3
from fastparquet import write
import pyarrow.parquet as pq
import numpy as np
import pandas as pd

from datasets import load_dataset
from CodeLanguageDetector.other_utils.files_io_utils import read_dict_pkl
from CodeLanguageDetector.data.utils import load_tokenizer


def get_language_dataset(
    languages_list_map_file: str,
    auth_token: str,
    destination_folder: str,
    max_number_of_samples: int,
    val_size: float,
    test_size: float
) -> None:
    """function for recieving raw data. As a result you can recieve 
    3 files of data for training, validattion and testing.

    Args:
        languages_list_map_file (str): code language map: language name - language id
        auth_token (str): auth token for huggingface 
        destination_folder (str): destination folder for saving recieved data
        max_number_of_samples (int): max number of code samples per language
        val_size (float): validation data share
        test_size (float): test data share

    Raises:
        e: Exception
    """
    languages_list_map = read_dict_pkl(languages_list_map_file)
    languages_list = list(languages_list_map.keys())

    row_id_train = 0
    row_id_val = 0
    row_id_test = 0
    
    train_file_name = os.path.join(
        destination_folder, f"data_{max_number_of_samples}_train.parquet"
    )
    val_file_name = os.path.join(
        destination_folder, f"data_{max_number_of_samples}_val.parquet"
    )
    test_file_name = os.path.join(
        destination_folder, f"data_{max_number_of_samples}_test.parquet"
    )
    
    if (
        os.path.exists(train_file_name)
        or os.path.exists(val_file_name)
        or os.path.exists(test_file_name)
    ):
        logger.info("Files already exist. Check, please ...")
        return
    
    for lang in tqdm(languages_list):
        logger.info(f"Loading data for {lang} language ...")
        
        lang_content_train = []
        lang_content_val = []
        lang_content_test = []

        lang_labels_train = []
        lang_labels_val = []
        lang_labels_test = []

        lang_ids_train = []
        lang_ids_val = []
        lang_ids_test = []

        idx = 0
        try:
            ds = load_dataset(
                "bigcode/the-stack",
                data_dir=f"data/{lang}",
                streaming=True,
                split="train",
                token=auth_token,
            )
        except Exception as e:
            logger.error(f"Error creating data iterator load_dataset. Exception: {e}")
            raise e
        
        for sample in tqdm(iter(ds)):
            spl_flg = np.random.choice(
                [0, 1, 2], p=[1 - val_size - test_size, val_size, test_size]
            )
            if spl_flg == 0:
                lang_content_train.append(sample["content"])
                lang_labels_train.append(languages_list_map[lang])
                lang_ids_train.append(row_id_train)
                row_id_train += 1
            elif spl_flg == 1:
                lang_content_val.append(sample["content"])
                lang_labels_val.append(languages_list_map[lang])
                lang_ids_val.append(row_id_val)
                row_id_val += 1
            else:
                lang_content_test.append(sample["content"])
                lang_labels_test.append(languages_list_map[lang])
                lang_ids_test.append(row_id_test)
                row_id_test += 1
            idx += 1
            if idx >= max_number_of_samples:
                break
        
        df_train = pd.DataFrame(
            {
                "id": lang_ids_train,
                "content": lang_content_train,
                "labels": lang_labels_train,
            }
        ).astype(
            {
                "id" : "int32",
                "content" : "object",
                "labels" : "int32"
            }
        )
        df_val = pd.DataFrame(
            {
                "id": lang_ids_val,
                "content": lang_content_val,
                "labels": lang_labels_val,
            }
        ).astype(
            {
                "id" : "int32",
                "content" : "object",
                "labels" : "int32"
            }
        )
        df_test = pd.DataFrame(
            {
                "id": lang_ids_test,
                "content": lang_content_test,
                "labels": lang_labels_test,
            }
        ).astype(
            {
                "id" : "int32",
                "content" : "object",
                "labels" : "int32"
            }
        )
        
        if not os.path.exists(train_file_name):
            try:
                df_train.to_parquet(train_file_name, engine="fastparquet")
                df_val.to_parquet(val_file_name, engine="fastparquet")
                df_test.to_parquet(test_file_name, engine="fastparquet")
                logger.info("Writing to files done successfully")
            except Exception as e:
                logger.error(f"Writing to files failed. Exception: {e}")
                raise e
        else:
            try:
                write(train_file_name, df_train, append=True)
                write(val_file_name, df_val, append=True)
                write(test_file_name, df_test, append=True)
                logger.info("Writing to files done successfully")
            except Exception as e:
                logger.error(f"Writing to files failed. Exception: {e}")
                raise e
        
        del df_train, df_val, df_test
        del lang_content_train, lang_content_val, lang_content_test
        del lang_labels_train, lang_labels_val, lang_labels_test
        del lang_ids_train, lang_ids_val, lang_ids_test

        logger.info(f"Loading data for {lang} language DONE")
        gc.collect()
    
    logger.info(f"Data loading FINISHED")
    gc.collect()


def string_processing(
    string: str,
    tokenizer: Any,
    max_length: int,
    unk_tokens_ratio: float,
    use_part_text: bool,
    cls_token_id: int,
    pad_token_id: int,
    unk_token_id: int,
    next_line_token: str,
) -> tuple:
    """Converting string into input ids and attention mask with defined
    tokenizer and conditions.

    Args:
        string (str): string for processing
        tokenizer (Any): tokenizer
        max_length (int): maximum count of tokens
        unk_tokens_ratio (float): token dropout power
        use_part_text (bool): using only random snippet from provided string
        next_line_token (str): next line token from tokenizer (used only if use_part_text=True)
        cls_token_id (int): special classification token id from tokenizer
        pad_token_id (int): padding token id from tokenizer
        unk_token_id (int): unknown token id from tokenizer

    Returns:
        tuple: input_ids, attention_mask of tokenized string
    """
    string_tokens = " ".join(tokenizer.tokenize(string))
    
    if use_part_text:
        str_lines = string_tokens.split(next_line_token)
        count_str_lines = len(str_lines)
        start_line_idx = randint(0, count_str_lines-1)
        end_line_idx = randint(start_line_idx + 1, count_str_lines)
        str_lines_selected = str_lines[start_line_idx:end_line_idx]
        string_tokens = next_line_token.join(str_lines_selected).strip()
    
    token_ids = tokenizer.convert_tokens_to_ids(string_tokens.split(" "))
    token_ids = [cls_token_id] + token_ids
    token_ids = token_ids[:max_length]
    input_ids = np.array(token_ids + (max_length - len(token_ids)) * [pad_token_id])
    
    input_attention_mask = (input_ids != pad_token_id) * 1

    if unk_tokens_ratio != 0.0:
        dropout_mask = np.random.choice(2, np.shape(input_ids), p=[1 - unk_tokens_ratio, unk_tokens_ratio])
        dropout_mask &= input_ids != pad_token_id
        dropout_mask &= input_ids != cls_token_id

        input_ids = np.choose(dropout_mask, [input_ids, np.full_like(input_ids, unk_token_id)])
    
    return input_ids, input_attention_mask

    
def get_processed_data(
    tokenizer_path: str,
    max_length: int,
    unk_tokens_ratio: float,
    use_part_text: bool,
    df_path: str,
    db_path: str,
    table_name: str,
    chunk_size: int,
) -> None:
    """function to get processed data.
    1. Reading parquet file with chunks
    2. Converting strings in raw data to input ids and attention mask
    3. Writing processed data to sqlite db

    Args:
        tokenizer_path (str): path for loading tokenizer
        max_length (int): maximum count of tokens
        unk_tokens_ratio (float): token dropout power
        use_part_text (bool): using only random snippet from provided string
        df_path (str): parquet df file path
        db_path (str): destination db file path
        table_name (str): db table name for writing processed data
        chunk_size (int): chunk size for reading parquet file

    Raises:
        e: Exception
    """
    tokenizer = load_tokenizer(tokenizer_path)
    next_line_token = tokenizer.tokenize("\n")[0]
    cls_token_id = tokenizer.cls_token_id
    pad_token_id = tokenizer.pad_token_id
    unk_token_id = tokenizer.unk_token_id
    
    columns = np.arange(0, max_length)
    columns_ids = [f"ids_{idx}" for idx in columns]
    columns_attention = [f"attention_{idx}" for idx in columns]
    columns_result = ["id", "labels"] + columns_ids + columns_attention
    columns_result_dtype = {col: "INTEGER PRIMARY KEY" if col == "id" else "INTEGER" for col in columns_result}
    
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        logger.info(f"Connected to db {db_path} successfully")
    except Exception as e:
        logger.error(f"Connection to db {db_path} failed. Exception: {e}")
        raise e
    
    try:
        cursor.execute("DROP TABLE IF EXISTS {}".format(table_name))
        connection.commit()
        logger.info(f"Dropped table {table_name} successfully")
    except Exception as e:
        logger.error(f"Dropping table {table_name} failed. Exception: {e}")
        raise e

    try:
        parquet_df_file = pq.ParquetFile(df_path)
        logger.info(f"Table {df_path} readed as parquet file successfully")
    except Exception as e:
        logger.error(f"Table {df_path} reading as parquet file failed. Exception: {e}")
        raise e
    
    logger.info(f"Processing data {df_path} STARTED")
    for batch in tqdm(parquet_df_file.iter_batches(batch_size=chunk_size)):
        batch_df = batch.to_pandas()
        input_ids_matr = []
        attention_matr = []
        for string in tqdm(batch_df["content"].values):
            input_ids, attention = string_processing(
                string=string,
                tokenizer=tokenizer,
                max_length=max_length,
                unk_tokens_ratio=unk_tokens_ratio,
                use_part_text=use_part_text,
                cls_token_id=cls_token_id,
                pad_token_id=pad_token_id,
                unk_token_id=unk_token_id,
                next_line_token=next_line_token,
            )
            input_ids_matr.append(input_ids)
            attention_matr.append(attention)
        input_ids_matr = np.array(input_ids_matr)
        attention_matr = np.array(attention_matr)
        df_input_ids = pd.DataFrame(data=input_ids_matr,columns=columns_ids)
        df_attention = pd.DataFrame(data=attention_matr,columns=columns_attention)
        df_result = pd.concat(
            [
                batch_df[["id", "labels"]],
                df_input_ids,
                df_attention
            ], axis=1
        )
        try:
            df_result[columns_result].to_sql(
                table_name,
                con=connection,
                dtype=columns_result_dtype,
                if_exists="append",
                index=False
            )
            logger.info(f"Chunck successfully added to {db_path} db at table {table_name}")
        except Exception as e:
            logger.error(f"Chunck adding to {db_path} db at table {table_name} failed. Exception: {e}")
            raise e
        del df_result, df_input_ids, df_attention
        del input_ids_matr, attention_matr
        
    logger.info(f"Processing data {df_path} FINISHED")
    gc.collect()
