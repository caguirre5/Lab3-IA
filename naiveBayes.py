import pandas as pd
import numpy as np


def naive_bayes(X_train, y_train, X_test):

    # Calcular las frecuencias de cada palabra en los mensajes de entrenamiento
    word_freq = {}
    for i in range(len(X_train)):
        for word in X_train[i].split():
            if word not in word_freq:
                word_freq[word] = [0, 0]
            word_freq[word][y_train[i]] += 1

    # Calcular las probabilidades a priori de cada categoría
    prior_prob = [np.count_nonzero(y_train == 0) / len(y_train),
                  np.count_nonzero(y_train == 1) / len(y_train)]

    # Clasificar los mensajes de prueba
    y_pred = []
    for text in X_test:
        # Calcular la probabilidad posterior de cada categoría para el mensaje
        post_prob = prior_prob.copy()
        for word in text.split():
            if word in word_freq:
                for i in range(2):
                    post_prob[i] *= (word_freq[word][i] + 1) / \
                        (sum(word_freq[word]) + len(word_freq))
        # Asignar la categoría con mayor probabilidad posterior
        y_pred.append(np.argmax(post_prob))

    return y_pred
