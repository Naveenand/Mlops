import sys

from heart_failure.exception import HeartFailureException
from heart_failure.logger import logging

from heart_failure.components.data_ingestion import DataIngestion
from heart_failure.components.data_validation import DataValidation
from heart_failure.components.data_transformation import DataTransformation
from heart_failure.components.model_trainer import ModelTrainer
from heart_failure.components.model_evaluation import ModelEvaluation
from heart_failure.components.model_pusher import ModelPusher

from heart_failure.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
   #DataTransformationConfig,
   #ModelTrainerConfig,
   #ModelEvaluationConfig,
   #ModelPusherConfig,
)

from heart_failure.entity.artifact_entity import (
    DataIngestionArtifact,
   #DataValidationArtifact,
   #DataTransformationArtifact,
   #ModelTrainerArtifact,
   #ModelEvaluationArtifact,
   #ModelPusherArtifact,
)

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
       #self.data_transformation_config = DataTransformationConfig()
       #self.model_trainer_config = ModelTrainerConfig()
       #self.model_evaluation_config = ModelEvaluationConfig()
       #self.model_pusher_config = ModelPusherConfig()

    # ======================================================
    # DATA INGESTION
    # ======================================================


    def start_data_ingestion(self) -> DataIngestionArtifact:
            """
            Starts the data ingestion component
            """
            try:
                logging.info(
                    "Entered start_data_ingestion method of TrainPipeline class"
                )

                logging.info("Fetching data from MongoDB")

                data_ingestion = DataIngestion(
                    data_ingestion_config=self.data_ingestion_config
                )

                data_ingestion_artifact = (
                    data_ingestion.initiate_data_ingestion()
                )

                logging.info(
                    "Successfully obtained train and test datasets"
                )

                logging.info(
                    "Exited start_data_ingestion method of TrainPipeline class"
                )

                return data_ingestion_artifact

            except Exception as e:
                logging.error(
                    "Error occurred in start_data_ingestion",
                    exc_info=True
                )
                raise HeartFailureException(e, sys) from e

    # ======================================================
    # DATA VALIDATION
    # ======================================================

    def start_data_validation(
        self,
        data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        """
        Starts the data validation component
        """
        logging.info(
            "Entered start_data_validation method of TrainPipeline class"
        )

        try:
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config,
            )

            data_validation_artifact = (
                data_validation.initiate_data_validation()
            )

            logging.info("Data validation completed successfully")

            logging.info(
                "Exited start_data_validation method of TrainPipeline class"
            )

            return data_validation_artifact

        except Exception as e:
            logging.error(
                "Error occurred in start_data_validation",
                exc_info=True
            )
            raise HeartFailureException(e, sys) from e