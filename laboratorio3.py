from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from naiveBayes import naive_bayes


def lab3(text):
    # Se convierten los textos a minusculas, se eliminan caracteres especiales y se crea un dataframe de pandas

    file = open('./entrenamiento.txt').readlines()
    types = []
    text = []

    for line in file:
        actual_type, actual_text = line[0:-1].split('\t')
        actual_text = str.lower(actual_text)
        actual_text = re.sub('[^a-zA-Z]', ' ', actual_text)
        actual_text = re.sub(r"\s+[a-zA-Z]\s+", ' ', actual_text)
        actual_text = re.sub(r'\s+', ' ', actual_text)
        types.append(actual_type)
        text.append(actual_text)

    data = pd.DataFrame({'types': types, 'text': text})

    # Se parametriza la data de spam y ham para tener valores de 0 y 1.
    # 0 = ham | 1 = spam

    encoder = LabelEncoder()
    data['types'] = encoder.fit_transform(data['types'])

    # Se dividen los datos para entreno y pruebas

    X_train, X_test, y_train, y_test = train_test_split(data['text'],
                                                        data['types'],
                                                        test_size=0.2,
                                                        random_state=0)

    # Task 1.2 - Construcción del modelo

    # Se convierte el texto en vectores de características

    data_train = pd.DataFrame({'type': y_train, 'text': X_train})
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(data_train['text'])

    X_train = X_train.tolist()
    y_train = y_train.tolist()
    X_test = X_test.tolist()

    # Entrenar el modelo usando Naive Bayes con los vectores de
    # características y las categorías de los datos de entrenamiento
    y_pred = naive_bayes(X_train, y_train, X_test)

    new_message = "Hey, Hey are Hey doing?"
    new_message_category = naive_bayes(X_train, y_train, [new_message])

    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)

    if new_message_category[0] == 0:
        return 'HAM', accuracy
    else:
        return 'SPAM', accuracy
