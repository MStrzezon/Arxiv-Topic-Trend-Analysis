import logging
from typing import Dict, Callable, Union, Optional

import pandas as pd
from top2vec import Top2Vec

from src.top2vec.utils import get_topvec_model_output, calculate_metrics

# Configure logging
logger = logging.getLogger('Top2VecMultiEvaluator')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class Top2VecMultiEvaluator:
    def __init__(self, documents, ngram_vocab=False, model_configs: Optional[Dict[str, Dict]] = None, octis_dataset=None, number_of_keywords=30,
                 models: Optional[Dict[str, Top2Vec]] = None):
        self.models = [] if models is None else models
        self.documents = documents
        self.model_configs = model_configs
        self.octis_dataset = octis_dataset
        self.number_of_keywords = number_of_keywords
        self.ngram_vocab = ngram_vocab

    def evaluate(self):
        if self.model_configs:
            for model_name, model_config in self.model_configs.items():
                if model_name not in self.models:
                    logger.info(f"Training model with embedding: {model_name}")
                    self.models[model_name] = Top2Vec(self.documents, embedding_model=model_config['embedding_model'], ngram_vocab=model_config['ngram_vocab'])

        if not self.models:
            raise ValueError("Either models or embeddings_models must be provided.")

        model_metrics = dict()
        for model_name, model in self.models.items():
            logger.info(f"Evaluating model metrics: {model_name}")
            model_metrics[model_name] = self.evaluate_model(model)
        return pd.DataFrame(model_metrics)

    def evaluate_model(self, model):
        topics_words = model.get_topics()[0][:, :self.number_of_keywords]
        topics_vectors = model.topic_vectors
        document_vectors = model.document_vectors
        model_output = get_topvec_model_output(self.documents, model.doc_top, topics_words, topics_vectors, document_vectors)
        model_metrics = calculate_metrics(model_output, self.octis_dataset)
        return model_metrics

