import re
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


# Download stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


# Text preprocessing function
def preprocess(text):
    text = re.sub(r'<.*?>', ' ', str(text))       # Remove HTML
    text = re.sub(r'http\S+|www\S+', ' ', text)   # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)      # Remove punctuation/digits
    text = text.lower()

    tokens = [
        stemmer.stem(w)
        for w in text.split()
        if w not in stop_words and len(w) > 2
    ]

    return ' '.join(tokens)


# 1. Load dataset
# Columns = [review_text, sentiment]
df = pd.read_csv('product_reviews.csv')

df['clean'] = df['review_text'].apply(preprocess)


# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['clean'],
    df['sentiment'],
    test_size=0.2,
    random_state=42,
    stratify=df['sentiment']
)


# 3. TF-IDF features
vec = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)

Xtr = vec.fit_transform(X_train)
Xte = vec.transform(X_test)


# 4. Train and evaluate three models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC()
}


for name, model in models.items():
    model.fit(Xtr, y_train)

    pred = model.predict(Xte)

    print(f"\n=== {name} ===")
    print("Accuracy:", round(accuracy_score(y_test, pred), 4))
    print(classification_report(y_test, pred))


# 5. Predict sentiment of a new review
best = models["Logistic Regression"]

sample = "The battery drains very fast, really disappointed with this product."

prediction = best.predict(
    vec.transform([preprocess(sample)])
)[0]

print("Prediction:", prediction)