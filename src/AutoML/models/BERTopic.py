from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity, InvertedRBO
from octis.evaluation_metrics.similarity_metrics import PairwiseJaccardSimilarity
from octis.evaluation_metrics.topic_significance_metrics import KL_uniform, KL_vacuous, KL_background
from bertopic import BERTopic

import pandas as pd
from .Model import Model


class CustomBERTopic(Model):
    def __init__(self, reduce_outliers: bool = True):
        self.reduce_outliers = reduce_outliers
        self.model = BERTopic(calculate_probabilities=True)

    def get_bertopic_model_output(self, topic_model: BERTopic) -> dict:
        model_output = dict()
        model_output['topics'] = topic_model.get_topic_info()['Representation'].tolist()
        model_output['topic-word-matrix'] = topic_model.c_tf_idf_.toarray()
        model_output['topic-document-matrix'] = topic_model.probabilities_
        return model_output

    def get_metrics(self, model_output, dataset, topk=10, verbose=False):
        umass = Coherence(texts=dataset.get_corpus(),
                          topk=topk, measure='u_mass')
        npmi = Coherence(texts=dataset.get_corpus(),
                         topk=topk, measure='c_npmi')
        c_v = Coherence(texts=dataset.get_corpus(), topk=topk, measure='c_v')
        topic_diversity = TopicDiversity(topk=topk)
        inv_rbo = InvertedRBO(topk=topk)
        pairwise_jaccard = PairwiseJaccardSimilarity()
        kl_uniform = KL_uniform()
        kl_vacuous = KL_vacuous()
        kl_background = KL_background()
        metrics = [(umass, 'UMass'), (npmi, 'NPMI'), (c_v, 'C_V'), (topic_diversity, 'Topic Diversity'),
                   (inv_rbo, 'Inverted RBO'),
                   (pairwise_jaccard, 'Pairwise Jaccard'), (kl_uniform, 'KL Uniform'), (kl_vacuous, 'KL Vacuous'),
                   (kl_background, 'KL Background')]
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

    def train_with_metrics_calculation(self, dataset) -> pd.DataFrame:
        data = []
        for doc in dataset.get_corpus():
            data.append(" ".join(doc))
        topics, probs = self.model.fit_transform(data)
        if self.reduce_outliers:
            new_topics = self.model.reduce_outliers(data, topics)
            self.model.update_topics(data, topics=new_topics)
        model_output = self.get_bertopic_model_output(self.model)
        result = self.get_metrics(model_output, dataset, verbose=False)
        results_df = pd.DataFrame([result])
        results_df.insert(0, 'model name', f'BERTopic - no_outliers:{self.reduce_outliers}')
        return results_df
    def __str__(self):
        return f"BERTopic - {self.reduce_outliers=}"
