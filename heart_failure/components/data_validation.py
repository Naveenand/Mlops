import json
import sys
import pandas as pd

from pandas import DataFrame
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection

from heart_failure.exception import HeartFailureException
from heart_failure.logger import logging
from heart_failure.utils.main_utils import read_yaml_file, write_yaml_file
from heart_failure.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact
)
from heart_failure.entity.config_entity import DataValidationConfig
from heart_failure.constants import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig
    ):
        """
        Data Validation for Heart Failure Dataset
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise HeartFailureException(e, sys)

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        """
        Validate total number of columns
        """
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"Required number of columns present: [{status}]")
            return status
        except Exception as e:
            raise HeartFailureException(e, sys)

    def is_column_exist(self, df: DataFrame) -> bool:
        """
        Validate numerical and categorical column existence
        """
        try:
            dataframe_columns = df.columns

            missing_numerical_columns = [
                col for col in self._schema_config["numerical_columns"]
                if col not in dataframe_columns
            ]

            missing_categorical_columns = [
                col for col in self._schema_config["categorical_columns"]
                if col not in dataframe_columns
            ]

            if missing_numerical_columns:
                logging.info(f"Missing numerical columns: {missing_numerical_columns}")

            if missing_categorical_columns:
                logging.info(f"Missing categorical columns: {missing_categorical_columns}")

            return not (missing_numerical_columns or missing_categorical_columns)

        except Exception as e:
            raise HeartFailureException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> DataFrame:
        """
        Read CSV file
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise HeartFailureException(e, sys)

    def detect_dataset_drift(
        self,
        reference_df: DataFrame,
        current_df: DataFrame
    ) -> bool:
        """
        Detect data drift using Evidently
        """
        try:
            data_drift_profile = Profile(
                sections=[DataDriftProfileSection()]
            )

            data_drift_profile.calculate(reference_df, current_df)
            report = json.loads(data_drift_profile.json())

            write_yaml_file(
                file_path=self.data_validation_config.drift_report_file_path,
                content=report
            )

            n_features = report["data_drift"]["data"]["metrics"]["n_features"]
            n_drifted_features = report["data_drift"]["data"]["metrics"]["n_drifted_features"]

            logging.info(f"Drift detected in {n_drifted_features}/{n_features} features")

            return report["data_drift"]["data"]["metrics"]["dataset_drift"]

        except Exception as e:
            raise HeartFailureException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Initiates full data validation process
        """
        try:
            logging.info("Starting Heart Dataset Data Validation")
            validation_error_msg = ""

            train_df = self.read_data(
                self.data_ingestion_artifact.trained_file_path
            )
            test_df = self.read_data(
                self.data_ingestion_artifact.test_file_path
            )

            if not self.validate_number_of_columns(train_df):
                validation_error_msg += "Training dataframe column mismatch. "

            if not self.validate_number_of_columns(test_df):
                validation_error_msg += "Test dataframe column mismatch. "

            if not self.is_column_exist(train_df):
                validation_error_msg += "Missing columns in training dataframe. "

            if not self.is_column_exist(test_df):
                validation_error_msg += "Missing columns in test dataframe. "

            validation_status = len(validation_error_msg) == 0

            if validation_status:
                drift_status = self.detect_dataset_drift(train_df, test_df)
                validation_error_msg = (
                    "Drift detected" if drift_status else "Drift not detected"
                )

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            logging.info(f"Data Validation Artifact: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise HeartFailureException(e, sys)
