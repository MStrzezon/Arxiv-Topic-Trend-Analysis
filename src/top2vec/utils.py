from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity, InvertedRBO
from octis.evaluation_metrics.similarity_metrics import PairwiseJaccardSimilarity
from octis.evaluation_metrics.topic_significance_metrics import KL_uniform, KL_vacuous, KL_background
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


def get_topvec_model_output(data, doc_top, topics, topic_vectors, document_vectors):
    topic_word_matrix = get_topic_word_matrix(data, doc_top)
    topic_document_matrix = get_topic_document_matrix(topic_vectors, document_vectors)

    model_output = dict()

    model_output['topics'] = topics
    model_output['topic-word-matrix'] = topic_word_matrix
    model_output['topic-document-matrix'] = topic_document_matrix

    return model_output


def get_topic_word_matrix(data, doc_top):
    processed_summaries = []
    for summary in data['Processed Summary']:
        processed_summaries.append(' '.join(summary))

    docs_df = pd.DataFrame(processed_summaries, columns=["Doc"])
    docs_df['Topic'] = doc_top
    docs_df['Doc_ID'] = range(len(docs_df))
    docs_per_topic = docs_df.groupby(['Topic'], as_index=False).agg({'Doc': ' '.join})
    tf_idf, _ = c_tf_idf(docs_per_topic.Doc.values, m=len(docs_per_topic.Doc.values))

    return tf_idf.T


def get_topic_document_matrix(topic_vectors, document_vectors):
    topic_document_matrix = cosine_similarity(document_vectors, topic_vectors)
    topic_document_matrix = (topic_document_matrix - topic_document_matrix.min()) / (
            topic_document_matrix.max() - topic_document_matrix.min())
    topic_document_matrix = topic_document_matrix / topic_document_matrix.sum(axis=1)[:, None]
    return topic_document_matrix.T


def c_tf_idf(documents, m, ngram_range=(1, 1)):
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    # number of rows where value is greater than 0
    sum_t = np.where(t > 0, 1, 0).sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count


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
