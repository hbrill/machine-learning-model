import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

data ="datasets/Suicide_Ideation_Dataset(Twitter-based).csv"

df = pd.read_csv(data)
df = df.dropna(subset=['Tweet', 'Label'])  # Drop rows with NaNs in these columns

# Assume 'text_column' contains the text data and 'label_column' contains the labels
X = df['Tweet']
y = df['Label']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train = X_train.dropna()
X_test = X_test.dropna()
y_train = y_train.dropna()
y_test = y_test.dropna()

# Create a pipeline with TF-IDF Vectorizer and Naive Bayes classifier
pipeline = make_pipeline(
    TfidfVectorizer(),
    MultinomialNB()
)

try:
    # Fit the model
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(["I'm so happy!"])

    # Evaluate the model
    print("Prediction: ", y_pred[0])
except Exception as e:
    print("Something went wrong!")