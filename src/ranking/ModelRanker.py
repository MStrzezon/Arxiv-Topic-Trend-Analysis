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


# TEST
if __name__ == "__main__":
    import pandas as pd

    data = {
        'model name': [
            'Doc2Vec + KMeans',
            'Doc2Vec + HDBSCAN [Processed data]',
            'Doc2Vec + HDBSCAN [Raw data]',
            'Doc2Vec + HDBSCAN [Processed data + ngrams]',
            'Doc2Vec + HDBSCAN [Raw data + ngrams]',
            'Allenai + Top2Vec [Raw data + ngrams]',
            'Allenai + Top2Vec [Processed data + ngrams]',
            'Allenai + Top2Vec [Processed data]',
            'Allenai + Top2Vec [Raw data]',
            'Bertopic with outliers',
            'Bertopic without outliers',
            'Best LDA'
        ],
        'UMass': [-2113, -4810, -5400, None, None, None, None, -9085, -8344, -6172, -5486, -3297.1],
        'NPMI': [0.138, 0.098, 0.077, None, None, None, None, -0.159, -0.125, 0.079, 0.094, -0.0294],
        'C_V': [0.691, 0.663, 0.617, None, None, None, None, 0.356, 0.381, 0.5669, 0.579, 0.3713],
        'Topic Diversity': [0.731, 0.563, 0.67, 0.903, 0.943, 0.916, 0.94, 0.845, 0.72, 0.646, 0.6158, 0.5836],
        'Inverted RBO': [0.986, 0.997, 0.998, 0.999, 0.999, 0.997, 0.995, 0.988, 0.987, 0.9963, 0.995, 0.8626],
        'Pairwise Jaccard': [0.0119, 0.0027, 0.002, 0.0004, 0.0003, 0.0023, 0.0045, 0.0105, 0.0102, 0.0027, 0.003, 0.1013],
        'KL Uniform': [2.872, 3.403, 3.303, 3.402, 3.302, 2.494, 2.336, 2.332, 2.506, None, None, 3.5118],
        'KL Vacuous': [9.262, 2.934, 2.928, 2.92, 2.921, 2.775, 2.685, 2.706, 2.8, None, None, 0.9419],
        'KL Background': [0.038, 0.019, 0.017, 0.013, 0.018, 0.011, 0.007, 0.006, 0.011, 1.3369, 1.3369, 0.4423]
    }

    df = pd.DataFrame(data)
    print(ModelRanker.rank_models(df))
    print("\n")
    print(ModelRanker.rank_models(df, top_n_numbers=3))
