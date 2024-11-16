import os
import sys
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from src.utils import load_object
from src.exception import CustomException
from src.logger import logging

class ModelEvaluation:
    def __init__(self):
        pass

    def eval_metrics(self, actual, pred):
        try:
            rmse = np.sqrt(mean_squared_error(actual, pred))  # RMSE
            mae = mean_absolute_error(actual, pred)  # MAE
            r2 = r2_score(actual, pred)  # R2 score
            return rmse, mae, r2
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_evaluation(self, train_array, test_array):
        try:
            # Split test data
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Load the trained model
            model_path = os.path.join("artifacts", "model.pkl")
            model = load_object(model_path)

            if not model:
                raise CustomException("Model could not be loaded.", sys)

            logging.info("Loaded model successfully for evaluation.")

            # Predict on test data
            predicted_qualities = model.predict(X_test)

            # Evaluate metrics
            rmse, mae, r2 = self.eval_metrics(y_test, predicted_qualities)
            logging.info(f"Evaluation Metrics - RMSE: {rmse}, MAE: {mae}, R2: {r2}")

            # Set MLflow tracking URI
            mlflow.set_registry_uri("http://127.0.0.1:5000")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # Create input example and infer signature
            input_example = X_test[:1]  # Use one row from test data as example
            signature = infer_signature(X_test, predicted_qualities)

            with mlflow.start_run():
                # Log metrics to MLflow
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)

                # Log the model to MLflow with input example and signature
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(
                        model,
                        artifact_path="model",
                        registered_model_name="CatBoostRegressor",
                        input_example=input_example,
                        signature=signature,
                    )
                else:
                    mlflow.sklearn.log_model(
                        model,
                        artifact_path="model",
                        input_example=input_example,
                        signature=signature,
                    )

        except Exception as e:
            raise CustomException(e, sys)
