import os
import sys
from typing import Any

import dill
import yaml
import numpy as np
from pandas import DataFrame

from heart_failure.exception import HeartFailureException
from heart_failure.logger import logger



# YAML FUNCTIONS


def read_yaml_file(file_path: str) -> dict:
    """
    Read YAML file and return contents as dictionary
    """
    try:
        logger.info(f"Reading YAML file: {file_path}")

        with open(file_path, "rb") as yaml_file:
            content = yaml.safe_load(yaml_file)

        logger.info("YAML file read successfully")
        return content

    except Exception as e:
        raise HeartFailureException(e, sys) from e


def write_yaml_file(file_path: str, content: Any, replace: bool = False) -> None:
    """
    Write content to a YAML file
    """
    try:
        logger.info(f"Writing YAML file: {file_path}")

        if replace and os.path.exists(file_path):
            os.remove(file_path)

        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "w") as file:
            yaml.dump(content, file, default_flow_style=False)

        logger.info("YAML file written successfully")

    except Exception as e:
        raise HeartFailureException(e, sys) from e



# OBJECT SERIALIZATION


def save_object(file_path: str, obj: object) -> None:
    """
    Save Python object using dill
    """
    logger.info("Entered save_object method")

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logger.info("Object saved successfully")

    except Exception as e:
        raise HeartFailureException(e, sys) from e


def load_object(file_path: str) -> object:
    """
    Load Python object using dill
    """
    logger.info("Entered load_object method")

    try:
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)

        logger.info("Object loaded successfully")
        return obj

    except Exception as e:
        raise HeartFailureException(e, sys) from e


# NUMPY FUNCTIONS


def save_numpy_array_data(file_path: str, array: np.array) -> None:
    """
    Save numpy array to file
    """
    try:
        logger.info(f"Saving numpy array: {file_path}")

        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)

        logger.info("Numpy array saved successfully")

    except Exception as e:
        raise HeartFailureException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """
    Load numpy array from file
    """
    try:
        logger.info(f"Loading numpy array: {file_path}")

        with open(file_path, "rb") as file_obj:
            array = np.load(file_obj)

        logger.info("Numpy array loaded successfully")
        return array

    except Exception as e:
        raise HeartFailureException(e, sys) from e


# DATAFRAME UTILITIES


def drop_columns(df: DataFrame, cols: list) -> DataFrame:
    """
    Drop columns from pandas DataFrame
    """
    logger.info("Entered drop_columns method")

    try:
        df = df.drop(columns=cols, axis=1)

        logger.info("Columns dropped successfully")
        return df

    except Exception as e:
        raise HeartFailureException(e, sys) from e
