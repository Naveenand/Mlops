import sys
import numpy as np
import pandas as pd

from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer

from heart_failure.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from heart_failure.entity.config_entity import DataTransformationConfig
from heart_failure.entity.artifact_entity import (
    DataTransformationArtifact,
    DataIngestionArtifact,
    DataValidationArtifact
)
from heart_failure.exception import HeartFailureException
from heart_failure.logger import logging
from heart_failure.utils.main_utils import (
    save_object,
    save_numpy_array_data,
    read_yaml_file,
    drop_columns
)


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise HeartFailureException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise HeartFailureException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates preprocessing pipeline for Heart Failure dataset
        """
        logging.info("Entered get_data_transformer_object method")

        try:
            num_features = self._schema_config["num_features"]
            transform_features = self._schema_config["transform_columns"]

            logging.info("Initializing transformers")

            numeric_scaler = StandardScaler()

            power_transformer = Pipeline(
                steps=[("power", PowerTransformer(method="yeo-johnson"))]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("power_transform", power_transformer, transform_features),
                    ("scaler", numeric_scaler, num_features)
                ],
                remainder="passthrough"
            )

            logging.info("Preprocessor pipeline created successfully")
            return preprocessor

        except Exception as e:
            raise HeartFailureException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Applies preprocessing + SMOTEENN and saves transformed artifacts
        """
        try:
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            logging.info("Starting data transformation")

            preprocessor = self.get_data_transformer_object()

            train_df = self.read_data(
                self.data_ingestion_artifact.trained_file_path
            )
            test_df = self.read_data(
                self.data_ingestion_artifact.test_file_path
            )

            logging.info("Train and test data loaded")

            X_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN]

            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN]

            drop_cols = self._schema_config.get("drop_columns", [])

            X_train = drop_columns(X_train, drop_cols)
            X_test = drop_columns(X_test, drop_cols)

            logging.info("Dropped unnecessary columns")

            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            logging.info("Applied preprocessing")

            smote = SMOTEENN(random_state=42)

            X_train_res, y_train_res = smote.fit_resample(
                X_train_arr, y_train
            )

            logging.info("Applied SMOTEENN on training data")

            train_arr = np.c_[X_train_res, y_train_res.to_numpy()]
            test_arr = np.c_[X_test_arr, y_test.to_numpy()]

            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                train_arr
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                test_arr
            )

            logging.info("Saved transformed artifacts")

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise HeartFailureException(e, sys)
