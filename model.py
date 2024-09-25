import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, LearningCurveDisplay, learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay

data ="dataset/clean_Suicide_Ideation_Dataset(Twitter-based).csv"
df = pd.read_csv(data)

df = df.dropna(subset=['Tweet', 'Label'])  # Drop rows with NaNs in the columns

# Assign X the Tweet column and y the Label column in the dataset
X = df['Tweet']
y = df['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Dropping NaN values from the sets above
X_train = X_train.dropna()
X_test = X_test.dropna()
y_train = y_train.dropna()
y_test = y_test.dropna()

# Creating a pipeline with TF-IDF Vectorizer and Naive Bayes classifier since we are working with strings
pipeline = make_pipeline(
    TfidfVectorizer(),
    MultinomialNB()
)

try:
    # Fit the model
    pipeline.fit(X_train, y_train)
except Exception as e:
    print(e)

# This method will make a prediction with test data, as well as display some visualizations
def visualize():
    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Calculate and display the confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)

    # Display a ROC Curve visualization
    disp2 = RocCurveDisplay.from_estimator(pipeline, X_test, y_test)

    # Display a Learning Curve Visualization
    train_sizes, train_scores, test_scores = learning_curve(pipeline, X, y)
    disp3 = LearningCurveDisplay(train_sizes=train_sizes, train_scores=train_scores, test_scores=test_scores, score_name="Score")
    
    disp1.plot()
    disp3.plot()
    plt.show()

def validate():
    y_pred = pipeline.predict(X_test)
    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

if __name__ == '__main__':
    validate()
    visualize()