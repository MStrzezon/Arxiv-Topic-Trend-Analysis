import logging

import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


from src.doc2vec.utils import calculate_metrics, c_tf_idf, extract_top_n_words_per_topic

logger = logging.getLogger('Doc2VecKMeansEvaluator')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class Doc2VecKMeansEvaluator:
    def __init__(self, documents, octis_dataset, doc2vec_args=None
                 , kmeans_args=None):
        if doc2vec_args is None:
            doc2vec_args = {
                'vector_size': 100,
                'window': 5,
                'min_count': 1,
                'workers': 4,
                'epochs': 20
            }
        if kmeans_args is None:
            kmeans_args = {
                'n_clusters': 35,
                'init': 'k-means++',
                'max_iter': 300,
                'n_init': 10,
                'random_state': 0
            }
        self.documents = documents
        self.tagged_data = [TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(documents)]
        self.doc2vec_args = doc2vec_args
        self.kmeans_args = kmeans_args
        self.octis_dataset = octis_dataset

    def evaluate(self):
        logger.info("Training Doc2Vec model")
        doc2vec_model = self.train_doc2vec()
        logger.info("Clustering abstracts")
        kmeans_model = self.train_kmeans(doc2vec_model)
        return doc2vec_model, kmeans_model, self.evaluate_model(doc2vec_model, kmeans_model)

    def train_doc2vec(self):
        model = Doc2Vec(self.tagged_data, **self.doc2vec_args)
        model.build_vocab(self.tagged_data)
        model.train(self.tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
        return model

    def train_kmeans(self, doc2vec_model):
        kmeans = KMeans(**self.kmeans_args)
        return kmeans.fit(doc2vec_model.dv.vectors)

    def evaluate_model(self, doc2vec_model, kmeans_model):
        docs_df = pd.DataFrame(self.documents, columns=["Doc"])
        docs_df['Topic'] = kmeans_model.labels_
        docs_df['Doc_ID'] = range(len(docs_df))
        docs_per_topic = docs_df.groupby(['Topic'], as_index=False).agg({'Doc': ' '.join})

        tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(self.documents))
        top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=30)
        top_n_words_list = [top_n_words[i] for i in range(len(top_n_words))]
        top_n_words_list = [[tuple[0] for tuple in topic] for topic in top_n_words_list]

        centroids = kmeans_model.cluster_centers_
        doc_embeddings = doc2vec_model.dv.vectors
        distances = cdist(doc_embeddings, centroids, 'euclidean')
        beta = 1.0
        weights = np.exp(-beta * distances.T)
        weights /= weights.sum(axis=0)

        model_output = dict()
        model_output['topics'] = top_n_words_list
        model_output['topic-word-matrix'] = tf_idf
        model_output["topic-document-matrix"] = weights

        model_metrics = calculate_metrics(model_output, self.octis_dataset)
        return pd.DataFrame([model_metrics])