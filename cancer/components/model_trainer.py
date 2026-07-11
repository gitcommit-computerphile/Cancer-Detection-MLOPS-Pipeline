import os
import sys
import mlflow
import numpy as np
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from cancer.entity.artifact_entity import (
    ClassificationMetricArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from cancer.entity.config_entity import ModelTrainerConfig
from cancer.exception.exception import CancerException
from cancer.logging.logger import logger
from cancer.utils.main_utils.utils import (
    evaluate_models,
    load_numpy_array_data,
    load_object,
    save_object,
)
from cancer.utils.ml_utils.metric.classification_metric import get_classification_score
from cancer.utils.ml_utils.model.estimator import BreastCancerModel

load_dotenv()


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact

    def _track_mlflow(
        self,
        best_model,
        train_metric: ClassificationMetricArtifact,
        test_metric: ClassificationMetricArtifact,
    ):
        with mlflow.start_run():
            mlflow.log_metric("train_f1", train_metric.f1_score)
            mlflow.log_metric("train_precision", train_metric.precision_score)
            mlflow.log_metric("train_recall", train_metric.recall_score)
            mlflow.log_metric("test_f1", test_metric.f1_score)
            mlflow.log_metric("test_precision", test_metric.precision_score)
            mlflow.log_metric("test_recall", test_metric.recall_score)
            mlflow.sklearn.log_model(best_model, "model")

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            models = {
                "RandomForest": RandomForestClassifier(random_state=42),
                "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
                "DecisionTree": DecisionTreeClassifier(random_state=42),
            }
            params = {
                "RandomForest": {"n_estimators": [50, 100], "max_depth": [None, 5]},
                "LogisticRegression": {"C": [0.1, 1.0]},
                "DecisionTree": {"max_depth": [None, 5, 10]},
            }

            report = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            logger.info(f"Model F1 scores: {report}")

            best_name = max(report, key=report.get)
            best_score = report[best_name]
            best_model = models[best_name]
            logger.info(f"Best model: {best_name} — test F1: {best_score:.4f}")

            if best_score < self.model_trainer_config.expected_score:
                raise Exception(
                    f"No model met the expected F1 of {self.model_trainer_config.expected_score}. "
                    f"Best was {best_name} at {best_score:.4f}"
                )

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_metric = get_classification_score(y_train, y_train_pred)
            test_metric = get_classification_score(y_test, y_test_pred)

            gap = abs(train_metric.f1_score - test_metric.f1_score)
            if gap > self.model_trainer_config.overfitting_underfitting_threshold:
                logger.warning(f"Train/test F1 gap {gap:.4f} exceeds threshold — possible overfit")

            preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)
            cancer_model = BreastCancerModel(preprocessor=preprocessor, model=best_model)

            self._track_mlflow(best_model, train_metric, test_metric)

            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            os.makedirs("final_model", exist_ok=True)
            save_object("final_model/model.pkl", cancer_model)
            save_object("final_model/preprocessor.pkl", preprocessor)

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric,
                test_metric_artifact=test_metric,
            )
        except Exception as e:
            raise CancerException(e, sys)