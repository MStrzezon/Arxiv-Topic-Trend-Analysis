from models.CustomLDA import CustomLDA
import pandas as pd
import time


class Scheduler:
    def __init__(self):
        self.models = [
            CustomLDA(25),
            CustomLDA(50),
        ]

    def train_all_models_with_metrics_calculation(self, dataset):
        results = []
        for model in self.models:
            print(f"Training model {model}")
            start_time = time.time()
            result = model.train_with_metrics_calculation(dataset)
            end_time = time.time()
            results.append(result)
            print(f"Training took {end_time - start_time} seconds")
            print("------------------------\n")
        return pd.concat(results)
