import os
import sys
import pickle
import numpy as np
import yaml
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

from cancer.exception.exception import CancerException
from cancer.logging.logger import logger


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise CancerException(e, sys)


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            yaml.dump(content, f)
    except Exception as e:
        raise CancerException(e, sys)


def save_numpy_array_data(file_path: str, array: np.ndarray):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            np.save(f, array)
    except Exception as e:
        raise CancerException(e, sys)


def load_numpy_array_data(file_path: str) -> np.ndarray:
    try:
        with open(file_path, "rb") as f:
            return np.load(f)
    except Exception as e:
        raise CancerException(e, sys)


def save_object(file_path: str, obj: object) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        raise CancerException(e, sys)


def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"File not found: {file_path}")
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise CancerException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models: dict, params: dict) -> dict:
    try:
        report = {}
        for name, model in models.items():
            param_grid = params.get(name, {})
            if param_grid:
                gs = GridSearchCV(model, param_grid, cv=3, scoring="f1", n_jobs=-1)
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = f1_score(y_test, y_pred)
            report[name] = score
            logger.info(f"  {name}: test F1 = {score:.4f}")
        return report
    except Exception as e:
        raise CancerException(e, sys)