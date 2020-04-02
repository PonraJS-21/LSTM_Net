# LSTM and CNN for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import load_model
from keras.preprocessing import sequence
import numpy as np

# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model

print(X_train[0])

model = load_model("lstm_imdb_review_model.h5")

index = 1

prediction = model.predict(np.expand_dims(X_test[index],axis=0))

print('Prediction: ', prediction)
print('Actual: ', y_test[index])