import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

data ="datasets/clean_Suicide_Ideation_Dataset(Twitter-based).csv"
df = pd.read_csv(data)
df = df.dropna(subset=['Tweet', 'Label'])  # Drop rows with NaNs in these columns

X = df['Tweet']
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def get_word_frequency():
    vectorizer = CountVectorizer()
    X_counts = vectorizer.fit_transform(X_train)
    word_counts = X_counts.sum(axis=0)
    word_freq = [(word, word_counts[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    word_freq.sort(key=lambda x: x[1], reverse=True)
    return word_freq[:10]

print("Most common words:", get_word_frequency())
