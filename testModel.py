import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
import pandas as pd
import numpy as np

data = pd.read_pickle("homeTeamStats.pickle")

print(data.head(5))



total_data_points = data.shape[0]
train_percentage = 0.7
train_data_points = int(total_data_points * train_percentage)



inputs =["blocked", "unblocked", "corner", "cross", "shotoff", "possession"]


test = data[train_data_points:]

test_data = np.asarray(test[inputs]).astype(np.float64)
test_label = np.asarray(test[["home_team_goal"]]).astype(np.uint8)


json_file = open("model_json.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
loaded_model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])




test_loss, test_acc = loaded_model.evaluate(test_data, test_label)


print("Test Accuracy: " + str(test_acc.item()))

predictions = loaded_model.predict(test_data)
print(predictions)
for i in range(20):
	print("The model guessed " + str(predictions[i].item()) + " and the answer was " + str(int(test_label[i])))