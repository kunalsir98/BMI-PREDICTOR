import sys 
import os 
from src.exception import CustomException
from src.logger import logging 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
import numpy as np
from sklearn.impute import SimpleImputer
import pandas as pd
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            numerical_cols = [
                'Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 
                'Resting_BPM', 'Session_Duration (hours)', 'Calories_Burned', 
                'Fat_Percentage', 'Water_Intake (liters)', 'Workout_Frequency (days/week)',
                'Experience_Level'
            ]
            categorical_cols = ['Gender', 'Workout_Type']

            logging.info('Reading numerical and categorical columns')

            # Pipeline for numerical features
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Pipeline for categorical features
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoding', OneHotEncoder(handle_unknown='ignore'))
            ])

            # Column transformer to apply pipelines
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])

            logging.info('Data transformation object created successfully')

            return preprocessor

        except Exception as e:
            logging.error('Exception occurred in get_data_transformation_obj')
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info(f'Train Columns : {train_df.columns}')
            logging.info(f'Test Columns:{test_df.columns}')

            logging.info('obtaining preprocessor file path')

            preprocessor_obj=self.get_data_transformation_obj()

            target_column_name='BMI'

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info('Applying preprocessor object on training data frame and testing dataframe')

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(
                self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            logging.info('Exception occured in datatransformation stage')
            raise CustomException(e, sys)



            


     
