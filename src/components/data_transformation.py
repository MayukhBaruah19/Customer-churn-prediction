'''
The main aim of the data_transformation file is to transform the data into a format suitable 
for model training, such as encoding categorical variables, scaling numerical features, 
and handling missing values,etc
'''

import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE #for handeling Imalanced data

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,handle_imbalanced_data



@dataclass
class DataTransformationConfig:
    """
    Configuration for data transformation.
    """
    preprocessor_obj_file_path=os.path.join('artifacts', 'preprocessor.pkl')
    label_encoder_obj_file_path = os.path.join('artifacts', 'label_encoder.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()    

    def get_data_transformer_object(self):
        '''
        This function is responsible for creating a data transformation pipeline, for numerical and categorical features.
        It handles missing values, scales numerical features, encodes categorical features, and prepares the data
        '''
        try:
            numerical_columns=['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

            categorical_columns=['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
             'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
             'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
             'PaperlessBilling', 'PaymentMethod'
             ]

        
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='mean')),
                    ('scaler',StandardScaler())
                ]
             )
            catergorical_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoding',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                    ]
            )
            
            logging.info(' Handeled missing values and scaled numerical features')
            logging.info('Handeled missing values and scaled categorical features')

            preprocessor=ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', catergorical_pipeline, categorical_columns)
                
            ]

            )
            logging.info('Created preprocessing object')

            return preprocessor


        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Read train and test data is completed')

            logging.info('Obtaining preprocessing object')


            preprocessor_obj=self.get_data_transformer_object()

            target_column_name='Churn'

            input_feature_train_df=train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df=test_df[target_column_name]

            # Label encode the target
            label_encoder = LabelEncoder()
            target_feature_train_df = label_encoder.fit_transform(target_feature_train_df)
            target_feature_test_df = label_encoder.transform(target_feature_test_df)

            logging.info('applying label encoding on target column both train and test data')
            
            
            # Transforming the input features
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
            
            logging.info('Applying preprocessing object on input columns of training  and test data ')
            
            #Applying SMOTE
            input_feature_train_resampled, target_feature_train_resampled = handle_imbalanced_data(
            input_feature_train_arr,
            target_feature_train_df
            )
            

            train_arr = np.c_[input_feature_train_resampled, np.array(target_feature_train_resampled)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f'saving preprocessing,and label encoder object')

            save_object(
                file_path=self.data_transformation_config.label_encoder_obj_file_path,
                obj=label_encoder
            )

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                self.data_transformation_config.label_encoder_obj_file_path,
                

            )

        except Exception as e:
            raise CustomException(e, sys)
            


