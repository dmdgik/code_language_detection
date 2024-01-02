from typing import Dict, Tuple
from loguru import logger
import pandas as pd
import numpy as np
import sqlite3
import gc

import torch
from torch import Tensor
from torch.utils.data import Dataset


class CodeLanguageDataset(Dataset):
    def __init__(
        self, 
        db_name: str, 
        db_table: str,
        in_memory: bool,
    ) -> None:
        """CodeLanguageDataset initialization

        Args:
            db_name (str): sqlite3 database path
            db_table (str): table name in databse
            in_memory (bool): load all data in memory for speed increasing (If you have enough RAM)

        Raises:
            e: Exception
        """
        super(CodeLanguageDataset, self).__init__()
        logger.info(f"CodeLanguageDataset initialization ...")
        try:
            self.conn = sqlite3.connect(db_name)
            self.db_table = db_table
            self.cur = self.conn.cursor()
            self.dataset_length = self.cur.execute(f"SELECT count(*) FROM {self.db_table}").fetchone()[0]
            self.in_memory = in_memory
            self.max_length = (len(self.cur.execute(f"SELECT * FROM {self.db_table} WHERE id=0").fetchone()) - 2) // 2
            self.cur.execute(f"SELECT * FROM {self.db_table} LIMIT 1")
            self.column_names = [i[0] for i in self.cur.description]
            if self.in_memory:
                query_for_all_data = f"SELECT * FROM {self.db_table}"
                chunks = pd.read_sql_query(query_for_all_data, self.conn, chunksize=20000)
                data_chunks = []
                for chunk in data_chunks:
                    chunks.append(chunk.astype({col_name : np.int32 for col_name in self.column_names}))
                self.data = pd.concat(chunks, ignore_index=True)
                for chunk in data_chunks:
                    del chunk
                del data_chunks
                gc.collect()
            logger.info(f"CodeLanguageDataset initialized successfully")
        except Exception as e:
            logger.error(f"CodeLanguageDataset initializing failed. Exception: {e}")
            raise e

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Tensor], Tensor]:
        raw = self.data.values[idx] if self.in_memory else self.cur.execute(f"SELECT * FROM {self.db_table} WHERE id={idx}").fetchone()
        label = raw[1]
        raw_id = raw[0]
        
        input_ids = raw[2:2+self.max_length]
        input_attention_mask = raw[2+self.max_length:2+2*self.max_length]

        return torch.tensor(raw_id, dtype=torch.int64), {
            "input_ids": torch.tensor(input_ids, dtype=torch.int32),
            "attention_mask": torch.tensor(input_attention_mask, dtype=torch.int32)
        }, torch.tensor(label, dtype=torch.int64)