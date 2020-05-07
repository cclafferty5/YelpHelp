import tensorflow as tf
import sys
from collections import Counter
import numpy as np
import json
import os

from keras.preprocessing.text import Tokenizer, tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model, Model
from keras.layers import Embedding, LSTM, Dense, Dropout, GRU, Bidirectional
from keras.callbacks import ModelCheckpoint
from keras.losses import SparseCategoricalCrossentropy, sparse_categorical_crossentropy, Loss, MSE
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras_bert import get_model, compile_model
from keras_bert.layers import Extract

cwd = os.getcwd()
models_dir = os.path.join(cwd, "models")
checkpoint_dir = os.path.join(models_dir, "checkpoints")

NUM_CLASSES = 5

global_indices = tf.constant([0., 1., 2., 3., 4.])
def star_squared_error(y_true, y_pred):
    indices = tf.reshape(tf.tile(global_indices, [tf.shape(y_pred)[0]]), tf.shape(y_pred))
    true_indices = tf.squeeze(y_true, axis=1)
    weighted = y_pred * indices
    weighted_avgs = tf.reduce_sum(weighted, axis=1)
    return (weighted_avgs - true_indices) ** 2

def hybrid_loss(weighting=[.5, .5]):
    entropy_weighting, error_weighting = weighting
    def loss_func(y_true, y_pred):
        entropy_loss = sparse_categorical_crossentropy(y_true, y_pred)
        star_loss = star_squared_error(y_true, y_pred)
        return entropy_weighting * entropy_loss + error_weighting * star_loss
    return loss_func

def build_model(input_length=300, rnn_size=256, loss='scc', use_glove=False, vocab_size=100000, learning_rate=1e-3, dropout_rate=.2, use_gru=False, use_bidirectional=False, use_c2v=False, show_accuracy=True):
    model = Sequential()
    if not use_c2v:    
        embed = Embedding(vocab_size, rnn_size, input_length=input_length)
        model.add(embed)
    if use_gru:
        rnn_cell = GRU(rnn_size, dropout=dropout_rate)
    else:
        rnn_cell = LSTM(rnn_size, dropout=dropout_rate)
    if use_bidirectional:
        model.add(Bidirectional(rnn_cell))
    else:
        model.add(rnn_cell)
    model.add(Dense(5, activation='softmax'))
    if loss == 'scc':
        loss_func = sparse_categorical_crossentropy
    elif loss == 'hybrid':
        loss_func = hybrid_loss
    optimizer = Adam(learning_rate=learning_rate)
    metrics = ['accuracy'] if show_accuracy else []
    model.compile(optimizer=optimizer, loss=loss_func, metrics=metrics)
    return model

def build_transformer_model(num_transformers=6, learning_rate=1e-3):
    inputs, output_layer = get_model(
        token_num=50000,
        head_num=5,
        transformer_num=num_transformers,
        embed_dim=100,
        feed_forward_dim=100,
        seq_len=300,
        pos_num=300,
        dropout_rate=0.05,
        training=False,
        trainable=True,
        output_layer_num=1
    )

    extract_layer = Extract(index=0, name='Extract')(output_layer)
    feed_forward_1 = Dense(units=100, name="feed_forward_1")(extract_layer)
    # feed_forward_2 = Dense(units=64, name="feed_forward_2")(feed_forward_1)
    output_logits = Dense(
        units=5,
        activation='softmax',
        name='NSP',
    )(feed_forward_1)

    model = Model(inputs, [output_logits])
    # for i, layer in enumerate(model.layers):
    #     # if i >= len(model.layers) - 3:
    #     if i != 2:
    #         layer.trainable = True
    # model.layers[2].set_weights([embedding_matrix])
    # model.layers[2].trainable = False
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_seqs, train_labels, num_epochs, save_as, batch_size=64, validation_split=.2):
    save_file = os.path.join(models_dir, save_as)
    checkpoint_file = os.path.join(checkpoint_dir, f"{save_as}.ckpt")
    cp_callback = ModelCheckpoint(filepath=checkpoint_file, verbose=1)
    training_result = model.fit(train_seqs, train_labels, epochs=num_epochs, batch_size=batch_size, validation_split=.2, callbacks=[cp_callback])
    model.save(save_file)

def load_keras_model(name, custom_objects={}, compile=True):
    return load_model(os.path.join(models_dir, name), custom_objects=custom_objects, compile=compile)

def load_transformer(name):
    weights_file = os.path.join(models_dir, name)
    model = build_transformer_model()
    model.load_weights(weights_file)
    return model

class YelpModel:
    def __init__(self, keras_model):
        self.keras_model = keras_model

    def predict_ratings(self, preprocessed_inputs):
        return [np.argmax(p) + 1 for p in self.keras_model.predict(preprocessed_inputs)]

class EnsembleModel(YelpModel):
    def __init__(self, config):
        self.num_models = len(config)
        self.models = [model for model, _ in config]
        self.weights = [weight for _, weight in config]
    
    def predict_ratings(self, preprocessed_inputs):
        assert len(preprocessed_inputs) == self.num_models
        num_samples = len(preprocessed_inputs[0])
        predictions = np.zeros((num_samples, NUM_CLASSES))
        for i, inputs in enumerate(preprocessed_inputs):
            predictions += self.weights[i] * self.models[i].predict(inputs)
        return [np.argmax(p) + 1 for p in predictions]

def load_custom_model(name, loss_func, custom_objects={}):
    model = load_keras_model(name, custom_objects=custom_objects, compile=False)
    model.compile(optimizer=Adam(), loss=loss_func)
    return model
        

####### MODELS ########

glove_gru_bi = load_keras_model("glove_gru_bi")
glove_gru_bi_char = load_keras_model("glove_gru_bi_char")
#transformer = load_transformer("bert_model_proper_glove_6.h5")
transformer = load_transformer("bert_model_6_5_extra_epochs.h5")
#gru_bi_50000 = load_keras_model("gru_bi_50000")
#gru_bi_50000_HL = load_custom_model("gru_bi_50000_hybrid_loss", hybrid_loss)
gru_bi_50000_star_loss = load_custom_model("gru_bi_50000_star_loss", star_squared_error)

GLOVE_GRU_BI = YelpModel(glove_gru_bi)
GLOVE_GRU_BI_CHAR = YelpModel(glove_gru_bi_char)
#GRU_BI_50000 = YelpModel(gru_bi_50000)
#GRU_BI_50000_HL = YelpModel(gru_bi_50000_HL)

ensemble_config = [(glove_gru_bi, .7), (glove_gru_bi_char, .3)]
CHAR_NO_CHAR_ENSEMBLE = EnsembleModel(ensemble_config)
GRU_BI_50000_STAR_LOSS = YelpModel(gru_bi_50000_star_loss)


TRANSFORMER = YelpModel(transformer)



#######################

BEST_MODEL = TRANSFORMER

