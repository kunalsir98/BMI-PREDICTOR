import sys
import os
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')
    raw_data_path = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion stage started')

        try:
            # Load data
            df = pd.read_csv(os.path.join('notebook', 'data.csv'))
            logging.info('DataFrame read as pandas DataFrame')

            # Ensure artifacts directory exists
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Split data into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.32)
            logging.info('Data split into train and test sets')

            # Save train and test data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Data Ingestion Completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error('Exception occurred in Data Ingestion Stage')
            raise CustomException(e, sys)

if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path,test_data_path = obj.initiate_data_ingestion()
    data_transformation=DataTransformation()
    train_arr,test_arr,_= data_transformation.initiate_data_transformation(train_data_path,test_data_path)

