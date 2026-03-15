import sys

from heart_failure.cloud_storage.aws_storage import SimpleStorageService
from heart_failure.exception import HeartFailureException
from heart_failure.logger import logging
from heart_failure.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from heart_failure.entity.config_entity import ModelPusherConfig
from heart_failure.entity.s3_estimator import HeartFailureEstimator


class ModelPusher:
    def __init__(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact,
        model_pusher_config: ModelPusherConfig
    ):
        """
        :param model_evaluation_artifact: Output reference of model evaluation stage
        :param model_pusher_config: Configuration for model pusher
        """

        self.s3 = SimpleStorageService()
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config

        self.heart_failure_estimator = HeartFailureEstimator(
            bucket_name=model_pusher_config.bucket_name,
            model_path=model_pusher_config.s3_model_key_path
        )

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        This method uploads the trained model to the S3 bucket
        """

        logging.info("Entered initiate_model_pusher method of ModelPusher class")

        try:
            logging.info("Uploading trained model to S3 bucket")

            self.heart_failure_estimator.save_model(
                from_file=self.model_evaluation_artifact.trained_model_path
            )

            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=self.model_pusher_config.s3_model_key_path
            )

            logging.info("Model uploaded successfully to S3")
            logging.info(f"Model pusher artifact: {model_pusher_artifact}")
            logging.info("Exited initiate_model_pusher method of ModelPusher class")

            return model_pusher_artifact

        except Exception as e:
            raise HeartFailureException(e, sys) from e