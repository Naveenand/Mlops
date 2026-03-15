import boto3
import os
import sys
import pickle
from io import StringIO
from typing import Union, List

from pandas import DataFrame, read_csv
from botocore.exceptions import ClientError
from mypy_boto3_s3.service_resource import Bucket

from heart_failure.configuration.aws_connection import S3Client
from heart_failure.logger import logging
from heart_failure.exception import HeartFailureException


class SimpleStorageService:

    def __init__(self):
        s3_client = S3Client()
        self.s3_resource = s3_client.s3_resource
        self.s3_client = s3_client.s3_client

    def s3_key_path_available(self, bucket_name: str, s3_key: str) -> bool:
        try:
            bucket = self.get_bucket(bucket_name)
            file_objects = [obj for obj in bucket.objects.filter(Prefix=s3_key)]
            return len(file_objects) > 0

        except Exception as e:
            raise HeartFailureException(e, sys)

    @staticmethod
    def read_object(object_name: str, decode: bool = True, make_readable: bool = False) -> Union[StringIO, str]:

        logging.info("Reading object from S3")

        try:
            func = (
                lambda: object_name.get()["Body"].read().decode()
                if decode else object_name.get()["Body"].read()
            )

            conv_func = lambda: StringIO(func()) if make_readable else func()

            return conv_func()

        except Exception as e:
            raise HeartFailureException(e, sys) from e

    def get_bucket(self, bucket_name: str) -> Bucket:

        logging.info("Fetching S3 bucket")

        try:
            bucket = self.s3_resource.Bucket(bucket_name)
            return bucket

        except Exception as e:
            raise HeartFailureException(e, sys) from e

    def get_file_object(self, filename: str, bucket_name: str) -> Union[List[object], object]:

        logging.info("Fetching file object from S3")

        try:
            bucket = self.get_bucket(bucket_name)

            file_objects = [obj for obj in bucket.objects.filter(Prefix=filename)]

            func = lambda x: x[0] if len(x) == 1 else x

            return func(file_objects)

        except Exception as e:
            raise HeartFailureException(e, sys) from e

    def load_model(self, model_name: str, bucket_name: str, model_dir: str = None) -> object:

        logging.info("Loading model from S3")

        try:
            model_file = model_name if model_dir is None else f"{model_dir}/{model_name}"

            file_object = self.get_file_object(model_file, bucket_name)

            model_obj = self.read_object(file_object, decode=False)

            model = pickle.loads(model_obj)

            return model

        except Exception as e:
            raise HeartFailureException(e, sys) from e

    def create_folder(self, folder_name: str, bucket_name: str) -> None:

        logging.info("Creating folder in S3")

        try:
            self.s3_resource.Object(bucket_name, folder_name).load()

        except ClientError as e:

            if e.response["Error"]["Code"] == "404":

                folder_obj = folder_name + "/"

                self.s3_client.put_object(
                    Bucket=bucket_name,
                    Key=folder_obj
                )

    def upload_file(self, from_filename: str, to_filename: str, bucket_name: str, remove: bool = True):

        logging.info(f"Uploading {from_filename} to S3 bucket {bucket_name}")

        try:

            self.s3_resource.meta.client.upload_file(
                from_filename,
                bucket_name,
                to_filename
            )

            if remove:
                os.remove(from_filename)
                logging.info("Local file removed after upload")

        except Exception as e:
            raise HeartFailureException(e, sys) from e

    def upload_df_as_csv(self, data_frame: DataFrame, local_filename: str, bucket_filename: str, bucket_name: str) -> None:

        logging.info("Uploading dataframe as CSV to S3")

        try:

            data_frame.to_csv(local_filename, index=False)

            self.upload_file(local_filename, bucket_filename, bucket_name)

        except Exception as e:
            raise HeartFailureException(e, sys) from e

    def get_df_from_object(self, object_: object) -> DataFrame:

        logging.info("Converting S3 object to dataframe")

        try:

            content = self.read_object(object_, make_readable=True)

            df = read_csv(content, na_values="na")

            return df

        except Exception as e:
            raise HeartFailureException(e, sys) from e

    def read_csv(self, filename: str, bucket_name: str) -> DataFrame:

        logging.info("Reading CSV from S3")

        try:

            csv_obj = self.get_file_object(filename, bucket_name)

            df = self.get_df_from_object(csv_obj)

            return df

        except Exception as e:
            raise HeartFailureException(e, sys) from e