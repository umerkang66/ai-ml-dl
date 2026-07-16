import numpy as np
import bentoml


@bentoml.service(resources={"cpu": "1"})
class IrisClassifierService:
    # Class-level model reference — BentoML loads this for you
    iris_model = bentoml.models.get("iris_classifier:latest")

    def __init__(self):
        # Load the actual sklearn model into memory
        self.model = bentoml.sklearn.load_model(self.iris_model)

    @bentoml.api
    def classify(self, input_data: np.ndarray) -> np.ndarray:
        return self.model.predict(input_data)
