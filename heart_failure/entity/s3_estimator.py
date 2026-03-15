from heart_failure.cloud_storage.aws_storage import SimpleStorageService
from heart_failure.exception import HeartFailureException
from heart_failure.entity.estimator import HeartFailureModel
import sys
from pandas import DataFrame


class HeartFailureEstimator:
    """
    This class is used to save and retrieve the Heart Failure model from the S3 bucket
    and perform predictions.
    """

    def __init__(self, bucket_name: str, model_path: str):
        """
        :param bucket_name: Name of the S3 bucket storing the model
        :param model_path: Path of the model inside the bucket
        """

        self.bucket_name = bucket_name
        self.s3 = SimpleStorageService()
        self.model_path = model_path
        self.loaded_model: HeartFailureModel = None

    def is_model_present(self, model_path: str) -> bool:
        """
        Check whether the model exists in the S3 bucket
        """
        try:
            return self.s3.s3_key_path_available(
                bucket_name=self.bucket_name,
                s3_key=model_path
            )
        except HeartFailureException as e:
            print(e)
            return False

    def load_model(self) -> HeartFailureModel:
        """
        Load the model from S3
        """
        return self.s3.load_model(
            self.model_path,
            bucket_name=self.bucket_name
        )

    def save_model(self, from_file: str, remove: bool = False) -> None:
        """
        Upload the model from local system to S3

        :param from_file: Local model file path
        :param remove: If True, delete the local file after upload
        """

        try:
            self.s3.upload_file(
                from_filename=from_file,
                to_filename=self.model_path,
                bucket_name=self.bucket_name,
                remove=remove
            )

        except Exception as e:
            raise HeartFailureException(e, sys)

    def predict(self, dataframe: DataFrame):
        """
        Make predictions using the trained model

        :param dataframe: Input dataframe
        :return: Prediction results
        """

        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()

            return self.loaded_model.predict(dataframe=dataframe)

        except Exception as e:
            raise HeartFailureException(e, sys)