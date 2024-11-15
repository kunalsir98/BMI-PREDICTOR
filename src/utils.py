import sys 
import os 
from src.exception import CustomException 
from src.logger import logging
import pickle
import pandas as pd 
import numpy as np 
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb")as file_obj:
            pickle.dump(obj,file_obj)


    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        report={}
        for model_name,model in models.items():
            model.fit(X_train,y_train)
            y_test_pred=model.predict(X_test)

            r2=r2_score(y_test,y_test_pred)
            report[model_name]=r2
        
        return report
    except Exception as e:                  
        logging.info('Exception occurred during model training')
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occurred in load_object function utils')
        raise CustomException(e, sys)    
    