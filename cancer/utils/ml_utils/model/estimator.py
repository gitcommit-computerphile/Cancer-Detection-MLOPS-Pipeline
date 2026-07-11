class BreastCancerModel:
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model

    def predict(self, x):
        transformed = self.preprocessor.transform(x)
        return self.model.predict(transformed)