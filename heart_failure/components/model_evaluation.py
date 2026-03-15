import sys
import os
import pandas as pd
from typing import Optional
from dataclasses import dataclass
from sklearn.metrics import f1_score

from heart_failure.entity.config_entity import ModelEvaluationConfig
from heart_failure.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataIngestionArtifact,
    ModelEvaluationArtifact
)

from heart_failure.exception import HeartFailureException
from heart_failure.logger import logging
from heart_failure.constants import TARGET_COLUMN
from heart_failure.entity.s3_estimator import HeartFailureEstimator


@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact
    ):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise HeartFailureException(e, sys) from e

    def get_best_model(self) -> Optional[HeartFailureEstimator]:
        """
        Retrieve the production model from S3 if it exists.
        If AWS credentials are not set, return None safely.
        """

        try:
            # Check if AWS credentials are set
            access_key = os.getenv("AWS_ACCESS_KEY_ID")
            secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

            if access_key is None or secret_key is None:
                logging.warning("AWS credentials not set. Skipping fetching best model from S3.")
                return None

            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path

            estimator = HeartFailureEstimator(
                bucket_name=bucket_name,
                model_path=model_path
            )

            if estimator.is_model_present(model_path=model_path):
                return estimator

            return None

        except Exception as e:
            raise HeartFailureException(e, sys)

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Compare the newly trained model with the production model
        """

        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            X = test_df.drop(TARGET_COLUMN, axis=1)
            y = test_df[TARGET_COLUMN]

            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score

            best_model_f1_score = None
            best_model = self.get_best_model()

            if best_model is not None:
                y_hat_best_model = best_model.predict(X)
                best_model_f1_score = f1_score(y, y_hat_best_model)

            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score

            result = EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                difference=trained_model_f1_score - tmp_best_model_score
            )

            logging.info(f"Model evaluation result: {result}")

            return result

        except Exception as e:
            raise HeartFailureException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Execute the full model evaluation pipeline
        """

        try:
            evaluation_response = self.evaluate_model()

            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluation_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluation_response.difference
            )

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")

            return model_evaluation_artifact

        except Exception as e:
            raise HeartFailureException(e, sys) from e