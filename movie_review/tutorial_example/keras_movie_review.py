import tensorflow as tf
from tensorflow import keras
import numpy

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# A dictionary mapping words to an integer index
_word_index = imdb.get_word_index()

word_index = {k:(v+3) for k,v in _word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
	return " ".join([reverse_word_index.get(i, "?") for i in text])

# this function will return the decoded (human readable) reviews  
    
''' If we have a look at some of our loaded in reviews we'll notice that they are different lengths. 
	This is an issue. We cannot pass different length data into out neural network. Therefore we must make each review the same length. 
	To do this we will follow the procedure below:
- if the review is greater than 250 words then trim off the extra words
- if the review is less than 250 words add the necessary amount of 's to make it equal to 250. '''

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)


# Defining the model

model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()

#Validation Data

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

''' If your targets are one-hot encoded, use categorical_crossentropy.
Examples of one-hot encodings:
[1,0,0]
[0,1,0]
[0,0,1]
But if your targets are integers, use sparse_categorical_crossentropy.
Examples of integer encodings (for the sake of completion):
1
2
3
But if your y_data is 0,1 then  use binary_crossentropy
 '''

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

results = model.evaluate(test_data, test_labels)
print(results)

model.save("imdb_review_model.h5") 

model = keras.models.load_model("imdb_review_model.h5")

''' To start we will need to integer encode the data. We will do this using the following function: '''

def review_encode(s):
	encoded = [1]

	for word in s:
		if word.lower() in word_index:
			encoded.append(word_index[word.lower()])
		else:
			encoded.append(2)

	return encoded

''' Next we will open our text file, read in each of the reviews (in this case just one) and use the model to predict whether it is positive or negative. '''

with open("The Lion King.txt", encoding="utf-8") as f:
	for line in f.readlines():
		nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"","").strip().split(" ")
		encode = review_encode(nline)
		encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250) # make the data 250 words long
		predict = model.predict(encode)
		print(line)
		print(encode)
		print(predict[0])