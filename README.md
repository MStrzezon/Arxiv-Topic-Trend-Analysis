# Arxiv Topic Trend Analysis

This project aims to analyze trends in Artificial Intelligence topic based on Arxiv articles' summaries using various machine learning models and techniques. The project includes data acquisition, preprocessing, exploratory data analysis (EDA), and topic modeling using different algorithms such as Doc2Vec, LDA, and BERTopic.

The results and insights from the analysis are presented in the project report.

## Project Structure

```
.gitignore
data/
src/
ai_topics_race.mp4 # Video of AI topics changes over time
report.pdf # Project report
README.md
```

### Directories

- **data**: Contains the raw and processed datasets used in the project.
- **src**: Contains the source code for different components of the project.
  - **`AutoML/`**: Contains scripts for automated machine learning and model ranking.
  - **`bertopic/`**: Contains scripts for topic modeling using BERTopic.
  - **`commons/`**: Contains common utility functions and scripts.
  - **`data-acquisition/`**: Contains scripts for acquiring and processing data.
  - **`doc2vec/`**: Contains scripts for topic modeling using Doc2Vec.
  - **`eda/`**: Contains scripts for exploratory data analysis.
  - **`LDA/`**: Contains scripts for topic modeling using LDA.
  - **`preprocessing/`**: Contains scripts for data preprocessing.
  - **`top2vec/`**: Contains scripts for topic modeling using Top2Vec.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/arxiv-topic-trend-analysis.git
   cd arxiv-topic-trend-analysis
   ```
