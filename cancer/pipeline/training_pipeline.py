import sys

from cancer.components.data_ingestion import DataIngestion
from cancer.components.data_transformation import DataTransformation
from cancer.components.data_validation import DataValidation
from cancer.components.model_trainer import ModelTrainer
from cancer.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
    ModelTrainerArtifact,
)
from cancer.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelTrainerConfig,
    TrainingPipelineConfig,
)
from cancer.exception.exception import CancerException
from cancer.logging.logger import logger
from cancer.constant.training_pipeline import TRAINING_BUCKET_NAME
from cancer.cloud.s3_syncer import S3Sync


class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        self.s3_sync = S3Sync()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            config = DataIngestionConfig(self.training_pipeline_config)
            logger.info("Starting data ingestion")
            artifact = DataIngestion(config).initiate_data_ingestion()
            logger.info(f"Data ingestion done: {artifact}")
            return artifact
        except Exception as e:
            raise CancerException(e, sys)

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            config = DataValidationConfig(self.training_pipeline_config)
            logger.info("Starting data validation")
            artifact = DataValidation(data_ingestion_artifact, config).initiate_data_validation()
            logger.info(f"Data validation done: {artifact}")
            return artifact
        except Exception as e:
            raise CancerException(e, sys)

    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            config = DataTransformationConfig(self.training_pipeline_config)
            logger.info("Starting data transformation")
            artifact = DataTransformation(data_validation_artifact, config).initiate_data_transformation()
            logger.info(f"Data transformation done: {artifact}")
            return artifact
        except Exception as e:
            raise CancerException(e, sys)

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            config = ModelTrainerConfig(self.training_pipeline_config)
            logger.info("Starting model training")
            artifact = ModelTrainer(config, data_transformation_artifact).initiate_model_trainer()
            logger.info(f"Model training done: {artifact}")
            return artifact
        except Exception as e:
            raise CancerException(e, sys)

    def sync_artifact_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(
                folder=self.training_pipeline_config.artifact_dir,
                aws_bucket_url=aws_bucket_url,
            )
        except Exception as e:
            raise CancerException(e, sys)

    def sync_saved_model_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/final_model"
            self.s3_sync.sync_folder_to_s3(folder="final_model", aws_bucket_url=aws_bucket_url)
        except Exception as e:
            raise CancerException(e, sys)

    def run_pipeline(self) -> ModelTrainerArtifact:
        try:
            ingestion_artifact = self.start_data_ingestion()
            validation_artifact = self.start_data_validation(ingestion_artifact)

            if not validation_artifact.validation_status:
                raise Exception("Data validation failed — check schema and drift report")

            transformation_artifact = self.start_data_transformation(validation_artifact)
            trainer_artifact = self.start_model_trainer(transformation_artifact)

            self.sync_artifact_dir_to_s3()
            self.sync_saved_model_dir_to_s3()

            return trainer_artifact
        except Exception as e:
            raise CancerException(e, sys)