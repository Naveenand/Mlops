import boto3
import os
from heart_failure.constants import (
    AWS_SECRET_ACCESS_KEY_ENV_KEY,
    AWS_ACCESS_KEY_ID_ENV_KEY,
    REGION_NAME
)


class S3Client:

    s3_client = None
    s3_resource = None

    def __init__(self, region_name: str = REGION_NAME):
        """
        This class retrieves AWS credentials from environment variables
        and establishes a connection with the S3 bucket.

        It raises an exception if the required environment variables are not set.
        """

        if S3Client.s3_resource is None or S3Client.s3_client is None:

            access_key_id = os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY)
            secret_access_key = os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY)

            if access_key_id is None:
                raise Exception(
                    f"Environment variable {AWS_ACCESS_KEY_ID_ENV_KEY} is not set."
                )

            if secret_access_key is None:
                raise Exception(
                    f"Environment variable {AWS_SECRET_ACCESS_KEY_ENV_KEY} is not set."
                )

            # Create S3 resource
            S3Client.s3_resource = boto3.resource(
                "s3",
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                region_name=region_name
            )

            # Create S3 client
            S3Client.s3_client = boto3.client(
                "s3",
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                region_name=region_name
            )

        self.s3_resource = S3Client.s3_resource
        self.s3_client = S3Client.s3_client