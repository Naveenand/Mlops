import sys

from pandas import DataFrame
from sklearn.pipeline import Pipeline

from heart_failure.exception import HeartFailureException
from heart_failure.logger import logging


# ===============================
# Target Mapping
# ===============================

class TargetValueMapping:
    def __init__(self):
        self.Alive: int = 0
        self.Death: int = 1

    def _asdict(self):
        return self.__dict__

    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))


# ===============================
# Model Wrapper
# ===============================

class HeartFailureModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        """
        :param preprocessing_object: Fitted preprocessing pipeline
        :param trained_model_object: Trained ML model
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: DataFrame):
        """
        Accepts raw input DataFrame,
        applies preprocessing,
        and returns model predictions
        """
        logging.info("Entered predict method of HeartFailureModel")

        try:
            logging.info("Applying preprocessing on input data")
            transformed_feature = self.preprocessing_object.transform(dataframe)

            logging.info("Generating predictions using trained model")
            predictions = self.trained_model_object.predict(transformed_feature)

            return predictions

        except Exception as e:
            raise HeartFailureException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"