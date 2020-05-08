import tensorflow as tf
import sys
from collections import Counter
import numpy as np
import json
import os

from keras.preprocessing.text import Tokenizer, tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, Dense, Dropout, GRU, Bidirectional
from keras.callbacks import ModelCheckpoint
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam

def batch_predict(batch, model, preprocessor):
    texts = [b["text"] for b in batch]
    batch_input = preprocessor.preprocess(texts)
    predictions = model.predict_ratings(batch_input)
    assert len(batch) == len(predictions)
    for i, b in enumerate(batch):
        b["predicted_stars"] = int(predictions[i])

def predict_test_set(test_set, model, preprocessor, batch_size=64, show_accuracy=True):
    for i in range(0, len(test_set), batch_size):
        batch = test_set[i: i + batch_size]
        batch_predict(batch, model, preprocessor)
    if show_accuracy:
        accuracy = (len([d for d in test_set if d["stars"] == d["predicted_stars"]]) / len(test_set)) * 100
        avg_star_error = sum([abs(d["predicted_stars"] - d["stars"]) for d in test_set]) / len(test_set)
        print("Accuracy: {:.3f}".format(accuracy))
        print("Average Star Error: {:.5f}".format(avg_star_error))
    
