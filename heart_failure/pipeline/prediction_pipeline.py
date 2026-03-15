import os
import sys
import pandas as pd
from pandas import DataFrame

from heart_failure.entity.config_entity import HeartFailurePredictorConfig
from heart_failure.entity.s3_estimator import HeartFailureEstimator
from heart_failure.exception import HeartFailureException
from heart_failure.logger import logging


class HeartFailureData:
    def __init__(
        self,
        age: float,
        anaemia: int,
        creatinine_phosphokinase: int,
        diabetes: int,
        ejection_fraction: int,
        high_blood_pressure: int,
        platelets: float,
        serum_creatinine: float,
        serum_sodium: int,
        sex: int,
        smoking: int,
        time: int
    ):
        """
        HeartFailureData constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.age = age
            self.anaemia = anaemia
            self.creatinine_phosphokinase = creatinine_phosphokinase
            self.diabetes = diabetes
            self.ejection_fraction = ejection_fraction
            self.high_blood_pressure = high_blood_pressure
            self.platelets = platelets
            self.serum_creatinine = serum_creatinine
            self.serum_sodium = serum_sodium
            self.sex = sex
            self.smoking = smoking
            self.time = time
        except Exception as e:
            raise HeartFailureException(e, sys) from e

    def get_heart_failure_input_data_frame(self) -> DataFrame:
        """
        Returns a DataFrame from HeartFailureData class input
        """
        try:
            data_dict = self.get_heart_failure_data_as_dict()
            return pd.DataFrame(data_dict)
        except Exception as e:
            raise HeartFailureException(e, sys) from e

    def get_heart_failure_data_as_dict(self) -> dict:
        """
        Returns a dictionary from HeartFailureData class input
        """
        logging.info("Entered get_heart_failure_data_as_dict method of HeartFailureData class")
        try:
            input_data = {
                "age": [self.age],
                "anaemia": [self.anaemia],
                "creatinine_phosphokinase": [self.creatinine_phosphokinase],
                "diabetes": [self.diabetes],
                "ejection_fraction": [self.ejection_fraction],
                "high_blood_pressure": [self.high_blood_pressure],
                "platelets": [self.platelets],
                "serum_creatinine": [self.serum_creatinine],
                "serum_sodium": [self.serum_sodium],
                "sex": [self.sex],
                "smoking": [self.smoking],
                "time": [self.time]
            }

            logging.info("Created heart failure input data dict")
            logging.info("Exited get_heart_failure_data_as_dict method")
            return input_data

        except Exception as e:
            raise HeartFailureException(e, sys) from e


class HeartFailureClassifier:
    def __init__(self, prediction_pipeline_config: HeartFailurePredictorConfig = HeartFailurePredictorConfig()):
        """
        :param prediction_pipeline_config: Configuration for prediction
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise HeartFailureException(e, sys) from e

    def predict(self, dataframe: DataFrame) -> str:
        """
        Predict heart failure outcome for the given input dataframe
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of HeartFailureClassifier class")

            model = HeartFailureEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path
            )

            prediction = model.predict(dataframe)
            return prediction

        except Exception as e:
            raise HeartFailureException(e, sys) from e