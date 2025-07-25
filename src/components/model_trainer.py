import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import accuracy_score, classification_report

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models



@dataclass
class ModelTrannerConfig():
        trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
        def __init__(self):
                self.model_tranner_config = ModelTrannerConfig()

        def initiate_model_trainer(self,train_array,test_array):
                try:
                    logging.info('Split training and test input data')
                    X_train,y_train,X_test,y_test=(
                        train_array[:,:-1], 
                        train_array[:,-1],
                        test_array[:,:-1],
                        test_array[:,-1]

                    )

                    models={
                            'LogisticRegression':LogisticRegression(),
                            'DecissionTree':DecisionTreeClassifier(),
                            'RandomForest':RandomForestClassifier(),
                             'Xgboost':XGBClassifier(),
                             'Catboost':CatBoostClassifier(verbose=0,random_state=42),
                             'SVM':SVC()
                    
                            }
                    

                    model_report,class_reports=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
                    logging.info(f"Model Report: {model_report}")
                    if model_report is None:
                             raise CustomException("Model evaluation returned None. Check evaluate_models().", sys)

                    # To get best model score and name from dictionary
                    best_model_name = max(model_report, key=model_report.get)
                    best_model_score = model_report[best_model_name]

                    # To get best model name from dictionary
                    best_model = models[best_model_name]

                    logging.info(f"Best Model Found , Model Name: {best_model_name} , Accuracy: {best_model_score}")

                    save_object(
                        file_path=self.model_tranner_config.trained_model_file_path,
                        obj=best_model
                    
                    )

                    predicted_output=best_model.predict(X_test)
                    accuracy=accuracy_score(y_test,predicted_output)
                    final_report = class_reports[best_model_name]
                    return accuracy,final_report

                except Exception as e:
                        raise CustomException(e, sys)                

        


        
