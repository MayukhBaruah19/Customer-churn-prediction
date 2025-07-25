'''
The main aim of the data_ingestion file is to read the data from some specific data source
(Like mongoDB,and other cloud storage etc) and split the data and then data_transformation will happend.
'''

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrannerConfig
from src.components.model_trainer import ModelTrainer


# Inputs that are required for data ingestion is present in this class and
# we will give the inputs through this class

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    # if my data is present in some database  to read the data from a
    def initiate_data_ingestion(self):
        # Database, we will write that code in this function.
        logging.info("Entered the data ingestion method or component")

        try:
            data = pd.read_csv('Notebook/data/cleaned_data.csv')
            logging.info('Read the dataset as dataframe from Local system')

            os.makedirs(os.path.dirname(
                self.ingestion_config.train_data_path), exist_ok=True)

            data.to_csv(self.ingestion_config.raw_data_path,
                        index=False, header=True)

            logging.info('Train test split initiated')
            train_set, test_set = train_test_split(
                data, test_size=0.2, random_state=42)

            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,
                            index=False, header=True)

            logging.info('Ingestion of the data is completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,

            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        # Step 1: Data Ingestion
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()

        # Step 2: Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

        # Step 3: Model Training
        model_trainer = ModelTrainer()
        accuracy, report = model_trainer.initiate_model_trainer(train_arr, test_arr)

        # Final Output
        print("\nFinal Model Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)

    except Exception as e:
        print(f"Pipeline failed due to: {e}")    



