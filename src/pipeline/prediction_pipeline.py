import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            # Load the preprocessor and model objects
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            # Transform the input features using the preprocessor
            data_scaled = preprocessor.transform(features)

            # Predict using the loaded model
            pred = model.predict(data_scaled)
            return pred
        
        except Exception as e:
            logging.error("Exception occurred during prediction")
            raise CustomException(e, sys)
class CustomData:
    def __init__(self,
                 Age: float,
                 Weight_kg: float,
                 Height_m: float,
                 Max_BPM: int,
                 Avg_BPM: int,
                 Resting_BPM: int,
                 Session_Duration_hours: float,
                 Calories_Burned: float,
                 Fat_Percentage: float,
                 Water_Intake_liters: float,
                 Workout_Frequency_days_week: int,
                 Experience_Level: int,
                 Gender: str,
                 Workout_Type: str):
        
        self.Age = Age
        self.Weight_kg = Weight_kg
        self.Height_m = Height_m
        self.Max_BPM = Max_BPM
        self.Avg_BPM = Avg_BPM
        self.Resting_BPM = Resting_BPM
        self.Session_Duration_hours = Session_Duration_hours
        self.Calories_Burned = Calories_Burned
        self.Fat_Percentage = Fat_Percentage
        self.Water_Intake_liters = Water_Intake_liters
        self.Workout_Frequency_days_week = Workout_Frequency_days_week
        self.Experience_Level = Experience_Level
        self.Gender = Gender
        self.Workout_Type = Workout_Type

    def get_data_as_dataframe(self):
        """
        Converts the custom data attributes into a pandas DataFrame for prediction.
        """
        try:
            # Convert feature dictionary to DataFrame
            custom_data_input_dict = {
                'Age': [self.Age],
                'Weight (kg)': [self.Weight_kg],
                'Height (m)': [self.Height_m],
                'Max_BPM': [self.Max_BPM],
                'Avg_BPM': [self.Avg_BPM],
                'Resting_BPM': [self.Resting_BPM],
                'Session_Duration (hours)': [self.Session_Duration_hours],
                'Calories_Burned': [self.Calories_Burned],
                'Fat_Percentage': [self.Fat_Percentage],
                'Water_Intake (liters)': [self.Water_Intake_liters],
                'Workout_Frequency (days/week)': [self.Workout_Frequency_days_week],
                'Experience_Level': [self.Experience_Level],
                'Gender': [self.Gender],
                'Workout_Type': [self.Workout_Type]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('DataFrame created successfully for prediction')
            return df
        except Exception as e:
            logging.error('Exception occurred while creating DataFrame')
            raise CustomException(e, sys)