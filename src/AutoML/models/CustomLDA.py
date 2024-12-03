from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity, InvertedRBO
from octis.evaluation_metrics.similarity_metrics import PairwiseJaccardSimilarity
from octis.evaluation_metrics.topic_significance_metrics import KL_uniform, KL_vacuous, KL_background
from octis.models.LDA import LDA
import pandas as pd
from .Model import Model


class CustomLDA(Model):
    def __init__(self, num_topics):
        self.num_topics = num_topics
        self.model = LDA(num_topics=num_topics)

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
        metrics = [(umass, 'UMass'), (npmi, 'NPMI'), (c_v, 'C_V'), (topic_diversity, 'Topic Diversity'), (inv_rbo, 'Inverted RBO'),
                   (pairwise_jaccard, 'Pairwise Jaccard'), (kl_uniform, 'KL Uniform'), (kl_vacuous, 'KL Vacuous'), (kl_background, 'KL Background')]
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
        output_model = self.model.train_model(dataset)
        result = self.get_metrics(output_model, dataset)
        result = pd.DataFrame([result])
        result.insert(0, 'model name', f'LDA - {self.num_topics}')
        return result

    def train(self, dataset):
        return self.model.train_model(dataset)
    
    def __str__(self):
        return f"LDA - {self.num_topics}"
