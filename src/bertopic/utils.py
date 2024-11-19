import pandas as pd
import os
import ast
import bertopic
import typing
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity, InvertedRBO
from octis.evaluation_metrics.similarity_metrics import PairwiseJaccardSimilarity
from octis.evaluation_metrics.topic_significance_metrics import KL_uniform, KL_vacuous, KL_background

os.environ["TOKENIZERS_PARALLELISM"] = 'true'


def parse_list_column(column):
    return ast.literal_eval(column)


def get_processed_data_bertopic(csv_file="../../data/arxiv_processed.csv"):
    df = pd.read_csv(csv_file, sep=";", converters={'Processed Summary': parse_list_column})
    df['Processed Summary'] = df['Processed Summary'].apply(lambda x: " ".join(x))
    return df


def get_bertopic_model_output(topic_model: bertopic.BERTopic) -> typing.Dict:
    model_output = dict()
    model_output['topics'] = topic_model.get_topic_info()['Representation'].tolist()
    model_output['topic-word-matrix'] = topic_model.c_tf_idf_
    model_output['topic-document-matrix'] = topic_model.probabilities_

    return model_output


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
    metrics = [(umass, 'UMass'), (npmi, 'NPMI'), (c_v, 'C_V'), (topic_diversity, 'Topic Diversity'), (inv_rbo, 'Inverted RBO'), (pairwise_jaccard, 'Pairwise Jaccard'), (kl_uniform, 'KL Uniform'), (kl_vacuous, 'KL Vacuous'), (kl_background, 'KL Background')]
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
