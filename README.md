## Sentiment Analysis with Various Models(only unigrams)

<div align="center">
  <a href="https://github.com/Sathvik1007/sentimental_classifier/stargazers"><img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/Sathvik1007/sentimental_classifier"></a>
  <a href="[https://github.com/mfts/papermark/blob/main/LICENSE](https://github.com/Sathvik1007/sentimental_classifier/blob/main/LICENSE)"><img alt="License" src="https://img.shields.io/badge/license-MIT-purple"></a>
</div>


This repository contains Python code for performing sentiment analysis on text data using different models, such as Naive Bayes, Support Vector Machines (SVM), and Word2Vec embeddings trained on 700 positive and 700 negative review data from source <a href = https://www.cs.cornell.edu/people/pabo/-movie-review-data/>DATA </a> or it is available <a href = https://github.com/Sathvik1007/sentimental_classifier/blob/main/review_polarity.tar.gz>here</a>. The analysis includes different representations of the text data, including word frequency, word occurrence, TF-IDF (Term Frequency-Inverse Document Frequency), and Word2Vec embeddings.<br />
**DONT FORGOT TO LEAVE A STAR** <br />

## Introduction
Sentiment analysis is a natural language processing task that involves determining the sentiment polarity of a given text, classifying it as positive or negative. This repository provides code to perform sentiment analysis using various models and text representations.

# Dependencies
**Python 3.x** <br />
**Libraries:** <br />
- **NumPy**
- **pandas**
- **sci-kit learn**
- **NLTK**
- **Gensim** (this is optional)
  
## Usage
To use this codebase, follow these steps:

**Clone the Repository:**

```bash
git clone https://github.com/Sathvik1007/sentimental_classifier.git
```


**Install Dependencies:**<br />
Ensure you have Python installed.
<br /><br />
Use pip to install required packages:
```bash
pip install nltk scikit-learn pandas gensim
```
<br /><br />
Run the Notebooks:
<br />
Open and run the Jupyter notebooks in the notebooks/ directory to understand each step of the sentiment analysis process.<br />
Each notebook focuses on a specific aspect of sentiment analysis, including data preprocessing, model implementation, and evaluation.<br />

## Data Preprocessing
The code implements various techniques for data preprocessing, including tokenization, creating word frequency tables, TF-IDF transformations, and Word2Vec embeddings. The code/preprocessing.py script contains functions for these preprocessing steps.

# Models Implemented
*Naive Bayes* <br />
Implementation of Multinomial Naive Bayes for sentiment analysis using word frequency and TF-IDF representations.<br />
*Support Vector Machines(SVM)* <br />
Implementation of SVM for sentiment analysis using word frequency, word occurrence, TF-IDF, and Word2Vec embeddings.<br />
*Word2Vec Embeddings* <br />
Word2Vec embeddings using Gensim to create word vectors for the text data.<br />
# Results
The results table provides the accuracy, precision, recall, and F1-score for each model <br />
```markdown
| Model                        | Accuracy | Precision | Recall | F1-Score | Support |
|------------------------------|----------|-----------|--------|----------|---------|
| Naive Bayes with Freq        | 0.708    | 0.71      | 0.70   | 0.71     | 300     |
| SVM using Freq               | 0.672    | 0.68      | 0.66   | 0.67     | 300     |
| Naive Bayes with Word Vector | 0.842    | 0.84      | 0.84   | 0.84     | 302     |
| MaxEnt with Word Vector      | 0.817    | 0.82      | 0.82   | 0.81     | 302     |
| SVM using Word Vector        | 0.787    | 0.78      | 0.79   | 0.79     | 302     |
| Naive Bayes with TF-IDF      | 0.833    | 0.83      | 0.83   | 0.83     | 302     |
| MaxEnt with TF-IDF           | 0.837    | 0.83      | 0.84   | 0.84     | 302     |
| SVM using TF-IDF             | 0.823    | 0.83      | 0.82   | 0.82     | 302     |
```
<br />
# Contributing

Contributions to improve this codebase are welcome! If you have suggestions, enhancements, or bug fixes, please create an issue or submit a pull request.
<br />
# License 
<br />
This project is licensed under the MIT License. See the LICENSE file for details.







