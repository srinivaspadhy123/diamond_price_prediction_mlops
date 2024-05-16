import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exceptions.exception import CustomException
from sklearn.model_selection import train_test_split
import os
import sys
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    raw_data_path:str = os.path.join("artifacts","raw.csv")
    train_data_path:str = os.path.join("artifacts","train.csv")
    test_data_path:str = os.path.join("artifacts","test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingesiton(self):
        logging.info("Data Ingestion Started")
        try:
            data = pd.read_csv("experiment/gemstone.csv")
            logging.info("Reading a Data Frame")

            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("Raw DataSet Saved to the directory")
            logging.info("Starting Train Test Split")
            train_data,test_data = train_test_split(data,test_size=0.25)
            logging.info("Train Test Split Completed")
            train_data.to_csv(self.ingestion_config.train_data_path,index=False)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False)
            logging.info("Data Ingestion Part Completed")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as exp:
            logging.info("Exception Occured during Data Ingestion")
            raise CustomException(exp,sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingesiton()
