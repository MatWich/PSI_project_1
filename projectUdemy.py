try:
    import tensorflow as tf
    from tensorflow import keras
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import sklearn
    from sklearn import *
    #from sklearn.preprocessing import scale, SimpleImputer
except ImportError:
    raise ImportError("Nie udalo sie zaimportowac wszystkich modulow")

df = pd.read_csv("udemy_output_All_Lifestyle_p1_p626.csv")
# daje to bo ma 7 roznych wartosci a nie 27k przez co acc moze byc wysokie, cos do 20 wartosci mysle ze by bylo spoko
target = 'num_published_practice_tests'

""" PRZYGOTOWANIE DANYCH """
le = preprocessing.LabelEncoder()

num_subscribers = le.fit_transform(list(df["num_subscribers"]))
avg_rating = le.fit_transform(list(df["avg_rating"]))
avg_rating_recent = le.fit_transform(list(df["avg_rating_recent"]))
rating = le.fit_transform(list(df["rating"]))
num_reviews = le.fit_transform(list(df["num_reviews"]))
is_wishlisted = le.fit_transform(list(df["is_wishlisted"]))
num_published_lectures = le.fit_transform(list(df["num_published_lectures"]))
num_published_practice_tests = le.fit_transform(list(df["num_published_practice_tests"]))
discount_price__amount = le.fit_transform(list(df["discount_price__amount"]))
price_detail__amount = le.fit_transform(list(df["price_detail__amount"]))

""" TYLKO DO y. X TRZEBA ZMIENIAC SAMEMU"""
dataDict = {"num_subscribers": num_subscribers, "avg_rating": avg_rating, "avg_rating_recent": avg_rating_recent,
            "rating": rating, "num_reviews": num_reviews,
            "is_wishlisted": is_wishlisted, "num_published_lectures": num_published_lectures,
            "num_published_practice_tests": num_published_practice_tests,
            "discount_price__amount": discount_price__amount, "price_detail__amount": price_detail__amount}

X = pd.DataFrame(zip(num_subscribers, avg_rating, avg_rating_recent, num_reviews, rating,
             price_detail__amount, discount_price__amount, num_published_lectures, is_wishlisted))

y = dataDict[target]
print(len(df[target].value_counts()))

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, stratify=y, test_size=0.1,
                                                                            random_state=43)

""" TWORZENIE MODELU"""
model = keras.Sequential()
model.add(keras.layers.Input(x_train.shape[1]))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(500, activation="relu"))  # kazdy neuron jest z kazdym polaczony
model.add(keras.layers.Dense(250, activation="relu"))  # kazdy neuron jest z kazdym polaczony
model.add(keras.layers.Dense(125, activation="relu"))  # kazdy neuron jest z kazdym polaczony
model.add(keras.layers.Dense(75, activation="relu"))  # kazdy neuron jest z kazdym polaczony
model.add(keras.layers.Dense(35, activation="sigmoid"))
model.add(keras.layers.Dense(len(df[target].value_counts()), activation="softmax"))

""" DUZA ACC I VAL_LOSS CALY CZAS"""
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=["accuracy"])

''' ROZNE ACC NISKA VAL_LOSS'''
model.compile(optimizer='adam',
              loss=tf.keras.losses.LogCosh(),
              metrics=['accuracy'])

''' WYJATKOWO SLABE ACC'''
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SquaredHinge(),
#               metrics=['accuracy'])


models = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))
print("MAX ACC: ", max(models.history["val_accuracy"]))
