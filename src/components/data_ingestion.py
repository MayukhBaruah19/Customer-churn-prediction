'''
The main aim of the data_ingestion file is to read the data from some specific data source
(Like mongoDB,and other cloud storage etc) and split the data and then data_transformation will happend.
'''

import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass



#Inputs that are required for data ingestion is present in this class and 
# we will give the inputs through this class

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):  #if my data is present in some database  to read the data from a 
                                      #Database, we will write that code in this function.
        logging.info("Entered the data ingestion method or component")

        try:
            data=pd.read_csv('Notebook\data\churn data.csv')
            logging.info('Read the dataset as dataframe from Local system')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            data.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Train test split initiated')
            train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)    

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion of the data is completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                
            )

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    obj1=DataIngestion()   
    obj1.initiate_data_ingestion()  #This will call the function and it will read the data from the csv file.
            
            
