import pandas as pd
from matplotlib import pyplot as plt


def plot_topics_trends(data, doc_top):
    data['Topic'] = doc_top
    data['Year'] = pd.to_datetime(data['Published']).dt.year
    topic_trend = data.groupby(['Year', 'Topic']).size().reset_index(name='Count')
    topic_trend_pivot = topic_trend.pivot(index='Year', columns='Topic', values='Count').fillna(0)
    y_min = topic_trend_pivot.min().min()
    y_max = topic_trend_pivot.max().max()

    number_of_topics = len(set(doc_top))

    fig, axs = plt.subplots(number_of_topics // 5 + 1, 5, figsize=(20, 5 * (number_of_topics // 5 + 1)))
    for i in range(number_of_topics):
        topic = topic_trend_pivot.columns[i]
        ax = axs[i // 5, i % 5]
        topic_trend_pivot[topic].plot(ax=ax, title=f'Topic {topic}', ylim=(y_min, y_max))

    plt.subplots_adjust(hspace=1)  # Adjust the height space between rows
    plt.show()


