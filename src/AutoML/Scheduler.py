# from models.CustomLDA import CustomLDA
# from models.BERTopic import CustomBERTopic
import pandas as pd
import time

from sentence_transformers import SentenceTransformer

from src.AutoML.models.BERTopic import CustomBERTopic
from src.AutoML.models.CustomLDA import CustomLDA
from src.AutoML.models.Doc2Vec import CustomDoc2Vec
from src.AutoML.models.Top2Vec import CustomTop2Vec

allenai_specter = SentenceTransformer('sentence-transformers/allenai-specter')
embedding_model = allenai_specter.encode

class Scheduler:
    def __init__(self):
        self.models = [
            CustomBERTopic(),
            CustomBERTopic(reduce_outliers=False),
            CustomLDA(25),
            CustomLDA(50),
            CustomDoc2Vec(),
            CustomTop2Vec({'embedding_model': embedding_model, 'ngram_vocab': False}),
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
