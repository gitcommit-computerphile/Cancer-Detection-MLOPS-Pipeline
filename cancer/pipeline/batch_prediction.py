import sys
import pandas as pd

from cancer.exception.exception import CancerException
from cancer.logging.logger import logger
from cancer.utils.main_utils.utils import load_object

MODEL_PATH = "final_model/model.pkl"


class BatchPredictionPipeline:
    def __init__(self, input_file_path: str):
        self.input_file_path = input_file_path

    def run_prediction(self) -> pd.DataFrame:
        try:
            model = load_object(MODEL_PATH)
            df = pd.read_csv(self.input_file_path)
            if "target" in df.columns:
                df = df.drop("target", axis=1)
            predictions = model.predict(df)
            df["predicted_target"] = predictions
            df["predicted_label"] = df["predicted_target"].map({0: "Malignant", 1: "Benign"})
            return df
        except Exception as e:
            raise CancerException(e, sys)