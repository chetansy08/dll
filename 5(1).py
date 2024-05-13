import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load IMDB dataset
max_features = 20000
maxlen = 80
batch_size = 32

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=5, validation_data=(x_test, y_test))

score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

print('Test score:', score)
print('Test accuracy:', acc)

text_example = "This movie was fantastic! I loved every moment of it."
# text_example = "I didn't find that movie enjoyable at all. Every moment seemed lacking, and it just didn't live up to my expectations."

# Tokenize the text
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts([text_example])
sequence = tokenizer.texts_to_sequences([text_example])

# Pad sequences
sequence = pad_sequences(sequence, maxlen=maxlen)

# Predict sentiment
prediction = model.predict(sequence)[0][0]

# Print the predicted sentiment
if prediction < 0.5:
    print("Negative sentiment")
else:
    print("Positive sentiment")
