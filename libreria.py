from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np


def modelo(msg):
    file = open('./entrenamiento.txt').readlines()
    types = []
    text = []

    for line in file:
        actual_type, actual_text = line[0:-1].split('\t')
        types.append(actual_type)
        text.append(actual_text)

    data = pd.DataFrame({'types': types, 'text': text})

    encoder = LabelEncoder()
    data['types'] = encoder.fit_transform(data['types'])
    data['types'].unique()

    X_train, X_test, y_train, y_test = train_test_split(data['text'],
                                                        data['types'],
                                                        test_size=0.3,
                                                        random_state=0)

    vectorizer = CountVectorizer(ngram_range=(1, 2)).fit(X_train)
    X_train_vectorized = vectorizer.transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    X_train_vectorized.toarray().shape

    model = MultinomialNB(alpha=1)
    model.fit(X_train_vectorized, y_train)

    y_pred = model.predict(X_test_vectorized)

    X_train_vectorized.toarray()

    new_message_vectorized = vectorizer.transform([msg])

    # Predecir la etiqueta de clase para el nuevo mensaje
    new_message_class = model.predict(new_message_vectorized)

    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)

    # Imprimir la etiqueta de clase predicha
    if new_message_class == 0:
        return 'HAM', accuracy
    else:
        return 'SPAM', accuracy
