# Import required libraries
import tensorflow as tf
import pandas as pd
import numpy as np
from transformers import BertTokenizer, TFBertForSequenceClassification

# Load the dataset
df = pd.read_csv("train_data.csv")

# Split the dataset into training and testing sets
train_size = int(0.8 * len(df))
train_data = df[:train_size]
test_data = df[train_size:]

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Encode the training and testing data
train_encodings = tokenizer(train_data["text"].tolist(), truncation=True, padding=True)
test_encodings = tokenizer(test_data["text"].tolist(), truncation=True, padding=True)

# Create TensorFlow datasets
train_dataset = (
    tf.data.Dataset.from_tensor_slices(
        (dict(train_encodings), train_data["label"].tolist())
    )
    .shuffle(10000)
    .batch(16)
)

test_dataset = tf.data.Dataset.from_tensor_slices(
    (dict(test_encodings), test_data["label"].tolist())
).batch(16)

# Load the BERT model for sequence classification
model = TFBertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_dataset, epochs=2, validation_data=test_dataset, batch_size=32)

# # Chat with the user to detect depression
# while True:
#     text = input("How are you feeling today? ")
#     encoding = tokenizer(text, truncation=True, padding=True, return_tensors='tf')
#     output = model(encoding)[0]
#     prediction = tf.argmax(output, axis=1)
#     if prediction == 1:
#         print("It seems like you might be feeling depressed. Please consider seeking help.")
#     else:
#         print("It's great to hear that you're doing well!")

# Save the model
model.save_pretrained("depression_bert_model.h5")
