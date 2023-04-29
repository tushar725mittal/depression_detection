import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# Load test data
test_data = pd.read_csv("test_data.csv")

# Load RNN model
rnn_model = tf.keras.models.load_model("depression_rnn_model.h5")

# Load BERT model
bert_model = tf.keras.models.load_model("depression_bert_model.h5")

# Evaluate RNN model on test data
rnn_predictions = rnn_model.predict(test_data["text"])
rnn_predictions = [1 if p >= 0.5 else 0 for p in rnn_predictions]
rnn_accuracy = accuracy_score(test_data["label"], rnn_predictions)
rnn_precision = precision_score(test_data["label"], rnn_predictions)
rnn_recall = recall_score(test_data["label"], rnn_predictions)
rnn_f1_score = f1_score(test_data["label"], rnn_predictions)
rnn_auc_roc = roc_auc_score(test_data["label"], rnn_predictions)

# Evaluate BERT model on test data
bert_predictions = bert_model.predict(test_data["text"])
bert_predictions = [1 if p >= 0.5 else 0 for p in bert_predictions]
bert_accuracy = accuracy_score(test_data["label"], bert_predictions)
bert_precision = precision_score(test_data["label"], bert_predictions)
bert_recall = recall_score(test_data["label"], bert_predictions)
bert_f1_score = f1_score(test_data["label"], bert_predictions)
bert_auc_roc = roc_auc_score(test_data["label"], bert_predictions)

# Print results
print("RNN Model Evaluation:")
print("Accuracy:", rnn_accuracy)
print("Precision:", rnn_precision)
print("Recall:", rnn_recall)
print("F1 Score:", rnn_f1_score)
print("AUC-ROC:", rnn_auc_roc)

print("BERT Model Evaluation:")
print("Accuracy:", bert_accuracy)
print("Precision:", bert_precision)
print("Recall:", bert_recall)
print("F1 Score:", bert_f1_score)
print("AUC-ROC:", bert_auc_roc)
