import pandas as pd


class ModelRanker:

    @staticmethod
    def rank_models(models_metrics: pd.DataFrame, top_n_numbers=None) -> pd.DataFrame:
        result = pd.DataFrame({'model name': models_metrics['model name']})

        for col in models_metrics.columns[1:]:
            result[col] = models_metrics[col].rank(
                ascending=False, method='min', na_option='bottom').astype(int)

        result['score'] = result.iloc[:, 1:].sum(axis=1)
        result = result.sort_values(
            by='score', ascending=True).reset_index(drop=True)

        if top_n_numbers:
            result = result.head(top_n_numbers)

        return result
