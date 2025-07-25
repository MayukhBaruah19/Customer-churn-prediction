import os
import sys
import numpy as np
import pandas as pd
import dill
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import accuracy_score,classification_report

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    """
    Save an object to a file using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(f"Error saving object: {e}")


def handle_imbalanced_data(X_train, y_train):
    try:
        smote=SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        logging.info("Data is balanced now")
        return X_train_resampled, y_train_resampled
    except Exception as e:
        logging.error("Error while applying SMOTE", exc_info=True)
        raise CustomException(e, sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models):
        try:
            report={}
            cls_report_dict={}
            for i in range(len(list(models))):
                model=list(models.values())[i]
                model.fit(X_train,y_train) #fit on train data

                y_train_pred=model.predict(X_train)
                y_test_pred=model.predict(X_test)

                train_model_score=accuracy_score(y_train,y_train_pred)
                test_model_score=accuracy_score(y_test,y_test_pred)

                train_cls_report=classification_report(y_train,y_train_pred)
                test_cls_report=classification_report(y_test,y_test_pred)

                logging.info(f"Train Model Score: {train_model_score}")
                logging.info(f"Test Model Score: {test_model_score}")

                logging.info(f"Train Model Classification Report: \n {train_cls_report}")
                logging.info(f"Test Model Classification Report: \n {test_cls_report}")

                report[list(models.keys())[i]] = test_model_score
                cls_report_dict[list(models.keys())[i]] = test_cls_report

            return report,cls_report_dict



        except Exception as e:
            logging.error("Error while evaluating models", exc_info=True)
            raise CustomException(e, sys)
        







    





