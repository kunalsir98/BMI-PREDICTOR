import numpy as np
import os
import sys
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_models(self, X_train, y_train, X_test, y_test, model):
        try:
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            r2 = r2_score(y_test, y_pred_test)

            return {
                "train_mse": train_mse,
                "test_mse": test_mse,
                "train_mae": train_mae,
                "test_mae": test_mae,
                "r2_score": r2
            }
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting dependent and independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Initialize the CatBoostRegressor model
            model = CatBoostRegressor(verbose=0)  # Set verbose=0 to suppress output during training

            # Train the model
            logging.info("Training the CatBoost Regressor model.")
            model.fit(X_train, y_train)

            # Evaluate the model
            metrics = self.evaluate_models(X_train, y_train, X_test, y_test, model)
            logging.info(f"Model Evaluation Metrics: {metrics}")

            # Save the trained model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            logging.info('CatBoost Regressor model trained and saved successfully.')

        except Exception as e:
            logging.error('Exception occurred during model training.')
            raise CustomException(e, sys)
