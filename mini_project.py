import re
from sklearn.feature_extraction.text import CountVectorizer
import string
from nltk.stem import PorterStemmer
import pandas as pd
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

tweets = pd.read_csv("Tweets.csv")
tweets.head()

print(tweets.shape)

print(tweets.columns)

tweets_df = tweets.drop(
    tweets[tweets["airline_sentiment_confidence"] < 0.5].index, axis=0
)
print(tweets_df.shape)

"""**Extracting the 'text' column (tweets) and 'airline_sentiment' column (sentiment labels)**"""

X = tweets_df["text"]
y = tweets_df["airline_sentiment"]

nltk.download("stopwords")
stop_words = stopwords.words("english")
punct = string.punctuation
stemmer = PorterStemmer()

cleaned_data = []
for i in range(len(X)):
    tweet = re.sub("[^a-zA-Z]", " ", X.iloc[i])
    tweet = tweet.lower().split()
    tweet = [
        stemmer.stem(word)
        for word in tweet
        if (word not in stop_words) and (word not in punct)
    ]
    tweet = " ".join(tweet)
    cleaned_data.append(tweet)

print(cleaned_data)

print(y)

sentiment_ordering = ["negative", "neutral", "positive"]

y = y.apply(lambda x: sentiment_ordering.index(x))
y.head()

cv = CountVectorizer(max_features=3000, stop_words=["virginamerica", "unit"])
X_fin = cv.fit_transform(cleaned_data).toarray()
X_fin.shape

model = MultinomialNB()
X_train, X_test, y_train, y_test = train_test_split(X_fin, y, test_size=0.3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

final_report = classification_report(y_test, y_pred)
print(final_report)


conf_matrix = confusion_matrix(y_test, y_pred)
labels = sentiment_ordering
plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    xticklabels=labels,
    yticklabels=labels,
    cmap="Blues",
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
