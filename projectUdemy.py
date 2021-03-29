try:
    import tensorflow as tf
    from tensorflow import keras
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import sklearn
    from sklearn import *
    import os
    #from sklearn.preprocessing import scale, SimpleImputer
except ImportError:
    raise ImportError("Nie udalo sie zaimportowac wszystkich modulow")

def create_model():
    """ TWORZENIE MODELU"""
    model = keras.Sequential()
    model.add(keras.layers.Input(x_train.shape[1]))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(500, activation="relu"))  # kazdy neuron jest z kazdym polaczony
    model.add(keras.layers.Dense(250, activation="relu"))  # kazdy neuron jest z kazdym polaczony
    model.add(keras.layers.Dense(125, activation="relu"))  # kazdy neuron jest z kazdym polaczony
    model.add(keras.layers.Dense(75, activation="relu"))  # kazdy neuron jest z kazdym polaczony
    model.add(keras.layers.Dense(35, activation="sigmoid"))
    model.add(keras.layers.Dropout(0.1))                    # zeby siec nie byla za bardzo dopasowana
    model.add(keras.layers.Dense(len(df[target].value_counts()), activation="softmax"))

    ''' ROZNE ACC NISKA VAL_LOSS'''
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.LogCosh(),
                  metrics=['accuracy'])

    ''' WYJATKOWO SLABE ACC'''
    # model.compile(optimizer='adam',
    #               loss=tf.keras.losses.SquaredHinge(),
    #               metrics=['accuracy'])

    """ DUZA ACC I VAL_LOSS CALY CZAS"""
    # model.compile(optimizer='adam',
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=["accuracy"])

    return model

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

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, stratify=y, test_size=0.2,
                                                                            random_state=43)



""" ZAPIS MODELI """
#checkPointPath = "training/cp-{epoch:04d}.ckpt"
checkPointPath = "training/cp-best.ckpt-logcosh"
checkpointDir = os.path.dirname(checkPointPath)

cpCallback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkPointPath,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
    )

""" TWORZENIE MODELU """
model = create_model()

""" TRENOWANIE MODELU (do zakomentowania gdy juz mamy zrobione)"""
models = model.fit(x_train, y_train, epochs=10, callbacks=[cpCallback], validation_data=(x_test, y_test), verbose=0)
#models = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))
print("MAX ACC: ", max(models.history["val_accuracy"]))


""" WCZYTYWANIE MODELU """
#checkPointPath = "training/cp-best.ckpt"      # nr epocha podac trzeba, aby dzialac zaczelo
model = create_model()
loss, acc = model.evaluate(x_test, y_test)
print(f"Nie trenowany model: loss: {loss}, acc: {acc}")
latest = tf.train.latest_checkpoint(checkpointDir)
model.load_weights(latest)
loss, acc = model.evaluate(x_test, y_test)
print(f"Przywrocony model loss, acc: {loss}, {acc}")
