# Tweet Sentiment Analysis

This project implements NLP pipeline for classifying the sentiment of tweets into positive, negative or neutral categories.
1. The first step is loading the raw tweet data from a CSV file into a Pandas DataFrame for ease of manipulation.
2. Crucial text preprocessing is then applied to transform the unstructured tweets into a suitable input format for machine learning.
   This involves key tasks like converting text to lowercase, tokenizing, removing non-alphabetic characters, eliminating stopwords and punctuation, and stemming words to their root form.
4. The sentiment labels are encoded into numeric categories to allow training a classifier.
5. For feature extraction from the preprocessed text, CountVectorizer is utilized to generate a bag-of-words representation of the tweets. The features are limited to the 3000 most frequently occurring words to reduce dimensionality.
6. A Multinomial Naive Bayes model is chosen for the classifier and trained on the extracted features and sentiment labels.
7. To evaluate model performance thoroughly, a separate test set is kept aside.
8. The trained model is applied on this test set and its predictions are analyzed using classification metrics like precision, recall and F1-score for each sentiment category.
9. Additionally, a confusion matrix visualization provides clear insight into correct and incorrect predictions for each class.

## Installation:

Requires:
- Python > 3.7

1. Clone the repository:

```
git clone https://github.com/gormeet/nlp-mini-project
```

2. Install Dependencies:

```
pip install -r requirments.txt
```

## Running

Either use python script or use Jupyter notebook to run the project

## Features

- Input: The input data consists of a CSV file named 'Tweets.csv,' which contains text data in the 'text' column and sentiment labels in the 'airline_sentiment' column. The data is loaded into a Pandas DataFrame.
- Output: The output of the code includes a Multinomial Naive Bayes sentiment classification model. The model's performance is evaluated using a classification report and a confusion  matrix, which provide metrics like accuracy, precision, recall, and F1-score.
- Source of text corpus and size: 
  The text corpus is derived from the 'Tweets.csv' file and the dataset is a collection of 14,640 tweets.
  Dataset Link: https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment
- Major Modules/Functionalities: 
  - Data Collection: This module will involve the acquisition of tweets using the Twitter API 
  - Text Pre-processing: Text pre-processing steps will be applied to clean and prepare the data. 
  - Feature Extraction: Transforming the pre-processed text data into numerical representations  that can be used for machine learning.  
  - Sentiment Analysis Model: Implementation of a deep learning model for sentiment analysis. 
  - Training the Model: The sentiment analysis model will be trained on a labelled dataset with  examples of tweets and their corresponding sentiment labels (positive, negative, or neutral). 
  - Sentiment Prediction: Applying the trained model to predict the sentiment of each tweet in  the input dataset. 
  - Model Evaluation: Assessing the model's performance using metrics like accuracy, precision,  recall, and F1 score. 
  - Output Visualization: Presenting the sentiment analysis results in a user-friendly format. 

- Text Pre-processing steps applied: 
  - Removal of non-alphabetic characters 
  - Conversion of text to lowercase 
  - Tokenization of text 
  - Removal of stopwords
  - Stemming of words
  - Removal of punctuation 

