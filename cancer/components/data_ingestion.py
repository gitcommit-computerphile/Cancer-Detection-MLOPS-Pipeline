import os
import sys
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.model_selection import train_test_split

from cancer.entity.artifact_entity import DataIngestionArtifact
from cancer.entity.config_entity import DataIngestionConfig
from cancer.exception.exception import CancerException
from cancer.logging.logger import logger

load_dotenv()
MONGO_DB_URL = os.getenv("MONGODB_URL_KEY")
if not MONGO_DB_URL:
    raise EnvironmentError("MONGODB_URL_KEY is not set. Add it to your .env file.")


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        try:
            client = MongoClient(MONGO_DB_URL)
            db = client[self.data_ingestion_config.database_name]
            collection = db[self.data_ingestion_config.collection_name]
            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns:
                df.drop("_id", axis=1, inplace=True)
            df.replace({"na": float("nan")}, inplace=True)
            return df
        except Exception as e:
            raise CancerException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(os.path.dirname(feature_store_file_path), exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        except Exception as e:
            raise CancerException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> DataIngestionArtifact:
        try:
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42,
                stratify=dataframe["target"],
            )
            os.makedirs(os.path.dirname(self.data_ingestion_config.training_file_path), exist_ok=True)
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            return DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )
        except Exception as e:
            raise CancerException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info("Fetching breast cancer data from MongoDB")
            dataframe = self.export_collection_as_dataframe()
            if dataframe.empty:
                raise Exception("MongoDB collection is empty — run push_data.py first")
            dataframe = self.export_data_into_feature_store(dataframe)
            logger.info(f"Feature store written: {dataframe.shape}")
            artifact = self.split_data_as_train_test(dataframe)
            logger.info(f"Data ingestion artifact: {artifact}")
            return artifact
        except Exception as e:
            raise CancerException(e, sys)