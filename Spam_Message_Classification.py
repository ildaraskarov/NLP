import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv('data/spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]

y = df['v1']
X = df['v2']


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    stratify=y,
                                                    random_state=0)

clf = Pipeline(
    [
        ('tfidf', TfidfVectorizer()),
        ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1,class_weight="balanced"))
    ]
)

clf.fit(X_train, y_train)

prediction = clf.predict(X_test)

print(classification_report(y_test, prediction))

print(confusion_matrix(y_test, prediction))


print("=======================================")

print(clf.predict([r'YOu are winner! Congratulations! You have won 500 USD from lottery']))

print(df['v2'].loc[df['v1'] == 'spam'].values[0])
