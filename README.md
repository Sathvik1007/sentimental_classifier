# Sentiment Analysis with Various Models(only unigrams)

<div align="center">
  <a href="https://github.com/Sathvik1007/sentimental_classifier/stargazers"><img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/mfts/papermark"></a>
  <a href="[https://github.com/mfts/papermark/blob/main/LICENSE](https://github.com/Sathvik1007/sentimental_classifier/blob/main/LICENSE)"><img alt="License" src="https://img.shields.io/badge/license-MIT-purple"></a>
</div>

This repository contains Python code for performing sentiment analysis on text data using different models, such as Naive Bayes, Support Vector Machines (SVM), and Word2Vec embeddings trained on 700 positive and 700 negative review data from source <a href = https://www.cs.cornell.edu/people/pabo/-movie-review-data/>DATA </a> or it is available <a href = https://github.com/Sathvik1007/sentimental_classifier/blob/main/review_polarity.tar.gz>here</a>. The analysis includes different representations of the text data, including word frequency, word occurrence, TF-IDF (Term Frequency-Inverse Document Frequency), and Word2Vec embeddings.

Table of Contents
Introduction
Usage
File Structure
Dependencies
Data Preprocessing
Models Implemented
Results
Contributing
License
Introduction
Sentiment analysis is a natural language processing task that involves determining the sentiment polarity of a given text, classifying it as positive or negative. This repository provides code to perform sentiment analysis using various models and text representations.

Usage
To use this codebase, follow these steps:

Clone the Repository:

bash
Copy code
git clone https://github.com/Sathvik1007/sentimental_classifier.git
Install Dependencies:
Ensure you have Python installed. Use pip to install required packages:

bash
Copy code
pip install -r requirements.txt
Run the Notebooks:

Open and run the Jupyter notebooks in the notebooks/ directory to understand each step of the sentiment analysis process.
Each notebook focuses on a specific aspect of sentiment analysis, including data preprocessing, model implementation, and evaluation.
Explore Code Files:

The code/ directory contains Python scripts for different models and data preprocessing techniques.
Execute the scripts to perform sentiment analysis on your own datasets.
File Structure
The repository is structured as follows:

code/: Contains Python scripts for different models and data preprocessing methods.
data/: Holds sample datasets used for sentiment analysis.
notebooks/: Jupyter notebooks explaining the step-by-step process.
README.md: Overview of the repository and instructions.
LICENSE: License information for the code.
Dependencies
Python 3.x
Libraries:
NumPy
pandas
scikit-learn
NLTK
Gensim
Data Preprocessing
The code implements various techniques for data preprocessing, including tokenization, creating word frequency tables, TF-IDF transformations, and Word2Vec embeddings. The code/preprocessing.py script contains functions for these preprocessing steps.

Models Implemented
Naive Bayes
Implementation of Multinomial Naive Bayes for sentiment analysis using word frequency and TF-IDF representations.
Support Vector Machines (SVM)
Implementation of SVM for sentiment analysis using word frequency, word occurrence, TF-IDF, and Word2Vec embeddings.
Word2Vec Embeddings
Word2Vec embeddings using Gensim to create word vectors for the text data.
Results
The results table provides the accuracy, precision, recall, and F1-score for each model and text representation. However, it lacks information regarding the dataset used and the scoring metrics. To ensure clarity and transparency, please provide details about the dataset and the specific scoring metrics used to generate these results.

Contributing
Contributions to improve this codebase are welcome! If you have suggestions, enhancements, or bug fixes, please create an issue or submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
