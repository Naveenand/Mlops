import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from heart_failure.entity.config_entity import DataIngestionConfig
from heart_failure.entity.artifact_entity import DataIngestionArtifact
from heart_failure.exception import HeartFailureException
from heart_failure.logger import logging
from heart_failure.data_access.heart_failure_data import HeartFailureData


class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig=DataIngestionConfig()):
        """
        :param data_ingestion_config: configuration for data ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise HeartFailureException(e,sys)
        

    
def export_data_into_feature_store(self) -> DataFrame:
    """
    Method Name : export_data_into_feature_store
    Description : This method exports data from MongoDB to a CSV file

    Output      : DataFrame
    On Failure  : Logs the error and raises HeartFailureException
    """
    try:
        logging.info("Exporting data from MongoDB")

        heart_failure_data = HeartFailureData()

        dataframe = heart_failure_data.export_collection_as_dataframe(
            collection_name=self.data_ingestion_config.collection_name
        )

        logging.info(f"Shape of dataframe: {dataframe.shape}")

        feature_store_file_path = self.data_ingestion_config.feature_store_file_path
        dir_path = os.path.dirname(feature_store_file_path)
        os.makedirs(dir_path, exist_ok=True)

        logging.info(
            f"Saving exported data into feature store at: {feature_store_file_path}"
        )

        dataframe.to_csv(
            feature_store_file_path,
            index=False,
            header=True
        )

        return dataframe

    except Exception as e:
        logging.error(
            "Failed to export data into feature store",
            exc_info=True
        )
        raise HeartFailureException(e, sys) from e
        

def split_data_as_train_test(self, dataframe: DataFrame) -> None:
    """
    Method Name : split_data_as_train_test
    Description : This method splits the dataframe into train and test sets
                  based on the configured split ratio.

    Output      : Train and test CSV files are created
    On Failure  : Logs the error and raises HeartFailureException
    """
    logging.info(
        "Entered split_data_as_train_test method of DataIngestion class"
    )

    try:
        train_set, test_set = train_test_split(
            dataframe,
            test_size=self.data_ingestion_config.train_test_split_ratio,
            random_state=42
        )

        logging.info("Performed train-test split on the dataframe")

        dir_path = os.path.dirname(
            self.data_ingestion_config.training_file_path
        )
        os.makedirs(dir_path, exist_ok=True)

        logging.info("Exporting train and test datasets to CSV files")

        train_set.to_csv(
            self.data_ingestion_config.training_file_path,
            index=False,
            header=True
        )

        test_set.to_csv(
            self.data_ingestion_config.testing_file_path,
            index=False,
            header=True
        )

        logging.info("Exported train and test datasets successfully")
        logging.info(
            "Exited split_data_as_train_test method of DataIngestion class"
        )

    except Exception as e:
        logging.error(
            "Error occurred during train-test split",
            exc_info=True
        )
        raise HeartFailureException(e, sys) from e


def initiate_data_ingestion(self) -> DataIngestionArtifact:
    """
    Method Name : initiate_data_ingestion
    Description : Initiates the data ingestion component of the
                  Heart Failure training pipeline.

    Output      : DataIngestionArtifact
    On Failure  : Logs the error and raises HeartFailureException
    """
    logging.info(
        "Entered initiate_data_ingestion method of DataIngestion class"
    )

    try:
        dataframe = self.export_data_into_feature_store()

        logging.info("Successfully fetched data from MongoDB")

        self.split_data_as_train_test(dataframe)

        logging.info("Train-test split completed")

        data_ingestion_artifact = DataIngestionArtifact(
            trained_file_path=self.data_ingestion_config.training_file_path,
            test_file_path=self.data_ingestion_config.testing_file_path
        )

        logging.info(
            f"Data ingestion artifact created: {data_ingestion_artifact}"
        )

        logging.info(
            "Exited initiate_data_ingestion method of DataIngestion class"
        )

        return data_ingestion_artifact

    except Exception as e:
        logging.error(
            "Error occurred during data ingestion",
            exc_info=True
        )
        raise HeartFailureException(e, sys) from e