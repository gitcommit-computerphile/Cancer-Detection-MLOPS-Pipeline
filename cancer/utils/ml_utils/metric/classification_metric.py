import sys
from sklearn.metrics import f1_score, precision_score, recall_score

from cancer.entity.artifact_entity import ClassificationMetricArtifact
from cancer.exception.exception import CancerException


def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    try:
        return ClassificationMetricArtifact(
            f1_score=f1_score(y_true, y_pred),
            precision_score=precision_score(y_true, y_pred),
            recall_score=recall_score(y_true, y_pred),
        )
    except Exception as e:
        raise CancerException(e, sys)