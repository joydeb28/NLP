import pandas as pd
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
import sys
sys.getdefaultencoding()

maxlen = 10
batch_size = 32
file_name = "input.csv"
word_dictionary = {}


# Load and prepare data of the spam.csv file.
def load_data(file_name):
    df = pd.read_csv(file_name)

    x_loaded = df.iloc[:,0]
    y_loaded = df.iloc[:,1]

    x_replaced = x_to_vect(x_loaded)
    y_replaced = y_to_vect(y_loaded)

    return x_replaced, y_replaced


# Creates a vector of int representations out of the sms message words.
def x_to_vect(x_loaded):
    all_words = []
    for content in x_loaded:
        content_str = beautify_word(str(content))
        content_words = content_str.split(" ")
        for content_word in content_words:
            all_words.append(content_word)

    for i in range(len(all_words)):
        word = all_words[i]
        if word not in word_dictionary:
            word_dictionary[word] = len(word_dictionary)

    x_replaced = []
    for content in x_loaded:
        content_str = beautify_word(str(content))
        content_words = content_str.split(" ")

        content_replaced = []
        for content_word in content_words:
            nr = word_dictionary[content_word]
            content_replaced.append(nr)

        x_replaced.append(content_replaced)

    return x_replaced


# Change classification strings to int representations.
def y_to_vect(y_loaded):
    y_replaced = []
    for classification in y_loaded:
        if classification == "yes":
            y_replaced.append(1)

        if classification == "no":
            y_replaced.append(0)

    return y_replaced


# Clean up words from unnecessary chars.
def beautify_word(word):
    return word.lower().replace(".", "").replace(",", "").replace("!", "").replace("?", "")


print('Loading data...')
x, y = load_data(file_name)

max_features = len(word_dictionary)

#max_features = 10

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=2, train_size=0.9)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.25, recurrent_dropout=0.25))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train model')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=40,
          validation_data=(x_test, y_test))

score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

print('Test score:', score)
print('Test accuracy:', acc)