import os
import sys
import pandas as pd
from scipy.stats import ks_2samp

from cancer.constant.training_pipeline import SCHEMA_FILE_PATH
from cancer.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from cancer.entity.config_entity import DataValidationConfig
from cancer.exception.exception import CancerException
from cancer.logging.logger import logger
from cancer.utils.main_utils.utils import read_yaml_file, write_yaml_file


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_config = data_validation_config
        self._schema = read_yaml_file(SCHEMA_FILE_PATH)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            expected = len(self._schema["columns"])
            actual = len(dataframe.columns)
            if actual != expected:
                logger.info(f"Column count mismatch: expected {expected}, got {actual}")
            return actual == expected
        except Exception as e:
            raise CancerException(e, sys)

    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame) -> bool:
        try:
            report = {}
            drift_found = False
            for col in base_df.columns:
                if col == "target":
                    continue
                stat, p_value = ks_2samp(base_df[col], current_df[col])
                drifted = p_value < 0.05
                if drifted:
                    drift_found = True
                report[col] = {"p_value": float(p_value), "drift_status": drifted}
            write_yaml_file(self.data_validation_config.drift_report_file_path, report, replace=True)
            return not drift_found
        except Exception as e:
            raise CancerException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact.trained_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            errors = []
            if not self.validate_number_of_columns(train_df):
                errors.append(f"Train: expected {len(self._schema['columns'])} columns, got {len(train_df.columns)}")
            if not self.validate_number_of_columns(test_df):
                errors.append(f"Test: expected {len(self._schema['columns'])} columns, got {len(test_df.columns)}")

            drift_ok = self.detect_dataset_drift(train_df, test_df)
            if not drift_ok:
                logger.warning(
                    "KS drift detected between train and test — logged to report.yaml. "
                    "Expected on small datasets due to high test sensitivity; not blocking."
                )

            validation_status = len(errors) == 0

            if not validation_status:
                logger.info(f"Validation errors: {errors}")
                os.makedirs(os.path.dirname(self.data_validation_config.invalid_train_file_path), exist_ok=True)
                train_df.to_csv(self.data_validation_config.invalid_train_file_path, index=False)
                test_df.to_csv(self.data_validation_config.invalid_test_file_path, index=False)
            else:
                os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path), exist_ok=True)
                train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False)
                test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False)

            return DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
        except Exception as e:
            raise CancerException(e, sys)