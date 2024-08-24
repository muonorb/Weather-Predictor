import joblib

class predict:
    def __init__(self, model_file):
        self.model = joblib.load(model_file)

    def value(self, input_data):
        pred = self.model.predict(input_data)
        return pred
