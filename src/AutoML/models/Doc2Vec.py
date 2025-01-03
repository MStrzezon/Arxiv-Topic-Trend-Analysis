import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import InvertedRBO, TopicDiversity
from octis.evaluation_metrics.similarity_metrics import PairwiseJaccardSimilarity
from octis.evaluation_metrics.topic_significance_metrics import KL_uniform, KL_vacuous, KL_background
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import cdist

from src.AutoML.models.Model import Model
from src.doc2vec.utils import c_tf_idf


def c_tf_idf(documents, m, ngram_range=(1, 1)):
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count


def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names_out()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in
                   enumerate(labels)}
    return top_n_words


def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['Topic'])
                   .Doc
                   .count()
                   .reset_index()
                   .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                   .sort_values("Size", ascending=False))
    return topic_sizes


def calculate_metrics(model_output, dataset, topk=10, verbose=False):
    umass = Coherence(texts=dataset.get_corpus(), topk=topk, measure='u_mass')
    npmi = Coherence(texts=dataset.get_corpus(), topk=topk, measure='c_npmi')
    c_v = Coherence(texts=dataset.get_corpus(), topk=topk, measure='c_v')
    topic_diversity = TopicDiversity(topk=topk)
    inv_rbo = InvertedRBO(topk=topk)
    pairwise_jaccard = PairwiseJaccardSimilarity()
    kl_uniform = KL_uniform()
    kl_vacuous = KL_vacuous()
    kl_background = KL_background()
    metrics = [(umass, 'UMass'), (npmi, 'NPMI'), (c_v, 'C_V'), (topic_diversity, 'Topic Diversity'),
               (inv_rbo, 'Inverted RBO'), (pairwise_jaccard, 'Pairwise Jaccard'), (kl_uniform, 'KL Uniform'),
               (kl_vacuous, 'KL Vacuous'), (kl_background, 'KL Background')]
    results = dict()
    for metric, name in metrics:
        try:
            metric_score = metric.score(model_output)
        except:
            metric_score = None
        finally:
            if verbose:
                print(f"{name}: {metric_score}")
            results[name] = metric_score
    return results


class CustomDoc2Vec(Model):
    def __init__(self, doc2vec_args=None, kmeans_args=None):
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
        self.doc2vec_args = doc2vec_args
        self.kmeans_args = kmeans_args

    def train_doc2vec(self, dataset):
        tagged_documents = [TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(dataset.get_corpus())]
        model = Doc2Vec(tagged_documents, **self.doc2vec_args)
        model.build_vocab(tagged_documents)
        model.train(tagged_documents, total_examples=model.corpus_count, epochs=model.epochs)
        return model

    def train_kmeans(self, doc2vec_model):
        kmeans = KMeans(**self.kmeans_args)
        return kmeans.fit(doc2vec_model.dv.vectors)

    def train_with_metrics_calculation(self, dataset) -> pd.DataFrame:
        data = []
        for doc in dataset.get_corpus():
            data.append(" ".join(doc))
        doc2vec_model = self.train_doc2vec(dataset)
        kmeans_model = self.train_kmeans(doc2vec_model)

        docs_df = pd.DataFrame(data, columns=["Doc"])
        docs_df['Topic'] = kmeans_model.labels_
        docs_df['Doc_ID'] = range(len(docs_df))
        docs_per_topic = docs_df.groupby(['Topic'], as_index=False).agg({'Doc': ' '.join})

        tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(data))
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

        model_metrics = calculate_metrics(model_output, dataset)
        results_df = pd.DataFrame([model_metrics])
        results_df.insert(0, 'model name', f'Doc2Vec-vector_size:{self.doc2vec_args["vector_size"]}, clusters:{self.kmeans_args["n_clusters"]}')
        return results_df
