import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from catboost import CatBoostRegressor
import sys
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.utils import save_object, evaluate_models
import mlflow
import mlflow.sklearn

class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_models(self, X_train, y_train, X_test, y_test, model):
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

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Initialize the CatBoostRegressor model
            model = CatBoostRegressor(verbose=0)  # Set verbose=0 to suppress output during training

            mlflow.set_experiment("First Experiment")

            mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Use this if running locally

            # Start MLflow run to log parameters and metrics
            with mlflow.start_run():
                # Log parameter for model type
                mlflow.log_param("Model Type", "CatBoost Regressor")

                # Evaluate the model
                model_report = self.evaluate_models(X_train, y_train, X_test, y_test, model)
                print(model_report)
                logging.info(f'Model Report: {model_report}')

                # Log metrics
                mlflow.log_metric("train_mse", model_report["train_mse"])
                mlflow.log_metric("test_mse", model_report["test_mse"])
                mlflow.log_metric("train_mae", model_report["train_mae"])
                mlflow.log_metric("test_mae", model_report["test_mae"])
                mlflow.log_metric("r2_score", model_report["r2_score"])

                # Log the trained model
                mlflow.sklearn.log_model(model, "CatBoostModel")

            # Save the trained model locally
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )
            
            print('CatBoost Regressor Model Trained and Saved successfully')
            logging.info('CatBoost Regressor Model Trained and Saved successfully')

        except Exception as e:
            logging.error('Exception occurred during Model Training')
            raise CustomException(e, sys)
