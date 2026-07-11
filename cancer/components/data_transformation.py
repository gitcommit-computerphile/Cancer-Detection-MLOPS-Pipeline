import sys
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cancer.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from cancer.entity.config_entity import DataTransformationConfig
from cancer.exception.exception import CancerException
from cancer.logging.logger import logger
from cancer.utils.main_utils.utils import save_numpy_array_data, save_object

TARGET_COLUMN = "target"


class DataTransformation:
    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig,
    ):
        self.data_validation_artifact = data_validation_artifact
        self.data_transformation_config = data_transformation_config

    @staticmethod
    def get_data_transformer_object() -> Pipeline:
        return Pipeline([
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler()),
        ])

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            train_df = pd.read_csv(self.data_validation_artifact.valid_train_file_path)
            test_df = pd.read_csv(self.data_validation_artifact.valid_test_file_path)

            X_train = train_df.drop(TARGET_COLUMN, axis=1)
            y_train = train_df[TARGET_COLUMN]
            X_test = test_df.drop(TARGET_COLUMN, axis=1)
            y_test = test_df[TARGET_COLUMN]

            preprocessor = self.get_data_transformer_object()
            preprocessor.fit(X_train)

            train_arr = np.c_[preprocessor.transform(X_train), np.array(y_train)]
            test_arr = np.c_[preprocessor.transform(X_test), np.array(y_test)]

            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)

            logger.info(f"Data transformation complete — train shape: {train_arr.shape}")
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
        except Exception as e:
            raise CancerException(e, sys)