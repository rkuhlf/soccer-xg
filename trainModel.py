import tensorflow as tf
from tensorflow import keras

import pandas as pd
import numpy as np

data = pd.read_pickle("homeTeamStats.pickle")


model = keras.Sequential([
	keras.layers.Input(shape=(6,)),
	keras.layers.Dense(1)
	])


inputs =["blocked", "unblocked", "corner", "cross", "shotoff", "possession"]


total_data_points = data.shape[0]
train_percentage = 0.7
train_data_points = int(total_data_points * train_percentage) # approximately 15000
train = data[:train_data_points]

# should be numpy.ndarray
train_data = np.asarray(train[inputs]).astype(np.float64)
train_label = np.asarray(train[["home_team_goal"]]).astype(np.uint8).reshape((train_data_points, 1))

keras.optimizers.Adam(lr=0.001)
model.compile(loss="mean_squared_error", metrics=["accuracy"])



model.fit(train_data, train_label, epochs=4)


model_json = model.to_json()
f = open("model_json.json", "w+")
f.write(model_json)
f.close()
model.save_weights("model.h5")

num = 10
to_print = model.predict(train_data[:num]).reshape((1, num))
to_print = np.vstack([to_print, train_label[:num].reshape((1, num))])

print(to_print.reshape((num, 2), order='F'))
