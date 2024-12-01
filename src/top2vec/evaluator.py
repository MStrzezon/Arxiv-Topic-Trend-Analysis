import pandas as pd
from top2vec import Top2Vec

from src.top2vec.utils import get_topvec_model_output, calculate_metrics


class Top2VecEvaluator:
    def __init__(self, documents, embedding_model, octis_dataset, model=None):
        self.model = model
        self.documents = documents
        self.embedding_model = embedding_model
        self.octis_dataset = octis_dataset

    def evaluate(self):
        if self.model is None:
            self.model = Top2Vec(self.documents, embedding_model=self.embedding_model)
        topics_words = self.model.get_topics()[0][:, :30]
        topics_vectors = self.model.topic_vectors
        document_vectors = self.model.document_vectors
        model_output = get_topvec_model_output(self.documents, self.model.doc_top, topics_words, topics_vectors, document_vectors)
        model_metrics = calculate_metrics(model_output, self.octis_dataset, verbose=True)
        return pd.DataFrame([model_metrics])