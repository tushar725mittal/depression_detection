import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd

# Load data from csv train_data.csv
train_data = pd.read_csv("train_data.csv")
# Prepare data
X = train_data.iloc[:, 0]  # input text data
y = train_data.iloc[:, 1]  # target depression labels (0 or 1)

# Tokenize data
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)

# Pad sequences
max_length = 100  # maximum sequence length
X_pad = tf.keras.preprocessing.sequence.pad_sequences(
    X_seq, maxlen=max_length, padding="post"
)

# Build model
model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1, 32, input_length=max_length))
model.add(LSTM(64))
model.add(Dense(1, activation="sigmoid"))

# Compile model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train model
model.fit(X_pad, y, epochs=10, batch_size=32)

# # Chat with user
# context = [] # initialize context
# while True:
#     user_input = input("You: ")
#     if user_input.lower() == 'quit':
#         break
#     context.append(user_input)
#     X_seq = tokenizer.texts_to_sequences([context[-max_length:]]) # limit context length to max_length
#     X_pad = tf.keras.preprocessing.sequence.pad_sequences(X_seq, maxlen=max_length, padding='post')
#     y_pred = model.predict(X_pad)[0][0] # predict depression label for the current context
#     if y_pred >= 0.5:
#         print("Bot: It seems like you might be feeling depressed.")
#     else:
#         print("Bot: It doesn't seem like you're feeling depressed.")

# Save model
model.save("depression_rnn_model.h5")
