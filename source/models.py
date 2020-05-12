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

def weighted_loss(loss_func, weights=[2., 5., 5., 3., 1.]):
    loss_weights = tf.constant(weights)
    def weighted_loss_func(y_true, y_pred):
        true_indices = tf.cast(tf.squeeze(y_true, axis=1), tf.int32)
        one_hots = tf.one_hot(true_indices, depth=5, dtype=tf.float32)
        weight_vec = tf.linalg.matvec(one_hots, loss_weights)
        return weight_vec * loss_func(y_true, y_pred)
    return weighted_loss_func


def hybrid_loss(weighting=[.5, .5]):
    entropy_weighting, error_weighting = weighting
    def loss_func(y_true, y_pred):
        entropy_loss = sparse_categorical_crossentropy(y_true, y_pred)
        star_loss = star_squared_error(y_true, y_pred)
        return entropy_weighting * entropy_loss + error_weighting * star_loss
    return loss_func


def build_model(input_length=150, rnn_size=256, loss='scc', use_glove=False, vocab_size=50000, 
                learning_rate=1e-3, dropout_rate=.2, use_gru=True, use_bidirectional=True, 
                use_c2v=False, show_accuracy=True, hybrid_weighting=[.5, .5]):
    model = Sequential()
    if not use_c2v:
        if use_glove:
            embed = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False)
        else:
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
    elif loss == 'star':
        loss_func = star_squared_error
    elif loss == 'hybrid':
        loss_func = hybrid_loss(hybrid_weighting)
    elif loss == 'wsl':
        loss_func = weighted_star_loss
    optimizer = Adam(learning_rate=learning_rate)
    metrics = ['sparse_categorical_accuracy'] if show_accuracy else []
    model.compile(optimizer=optimizer, loss=loss_func, metrics=metrics)
    return model

def build_char_model(input_length=150, word_length=5, word_embedding_dim=100, 
                     char_embedding_dim=10, use_glove=False, vocab_size=50000, 
                     char_vocab_size=72, learning_rate=1e-3, dropout_rate=.2, 
                     use_gru=True, use_bidirectional=True, use_c2v=False, loss='scc',
                     show_accuracy=True, weight_loss=False, loss_weights=[2., 5., 5., 3., 1.]):
    word_inputs = Input(shape=(input_length,))
    char_inputs = Input(shape=(input_length, word_length))
    flattened_chars = Flatten()(char_inputs)
    if not use_c2v:
        if use_glove:
            embed = Embedding(vocab_size, word_embedding_dim, weights=[embedding_matrix], trainable=False)
        else:
            embed = Embedding(vocab_size, word_embedding_dim, input_length=input_length)
        word_embeddings = embed(word_inputs)
        flattened_character_embeddings = Embedding(char_vocab_size, char_embedding_dim, input_length=word_length * input_length)(flattened_chars)
        character_embeddings = Reshape((input_length, word_length * char_embedding_dim))(flattened_character_embeddings)
    embeddings = Concatenate()([word_embeddings, character_embeddings])
    rnn_size = word_embedding_dim + word_length * char_embedding_dim
    if use_gru:
        rnn_cell = GRU(rnn_size, dropout=dropout_rate)
    else:
        rnn_cell = LSTM(rnn_size, dropout=dropout_rate)
    if use_bidirectional:
        rnn_out = Bidirectional(rnn_cell)(embeddings)
    else:
        rnn_out = rnn_cell(embeddings)
    logits = Dense(5, activation='softmax')(rnn_out)
    model = Model([word_inputs, char_inputs], logits)
    if loss == 'scc':
        loss_func = sparse_categorical_crossentropy
    elif loss == 'star':
        loss_func = star_squared_error
    elif loss == 'hybrid':
        loss_func = hybrid_loss(hybrid_weighting)
    if weight_loss:
        loss_func = weighted_loss(loss_func, loss_weights)
    optimizer = Adam(learning_rate=learning_rate)
    metrics = ['sparse_categorical_accuracy'] if show_accuracy else []
    model.compile(optimizer=optimizer, loss=loss_func, metrics=metrics)
    return model

def build_transformer_model(num_transformers=6, learning_rate=1e-3):
    def weighted_loss(loss_func, weights=[2., 5., 5., 3., 1.]):
        loss_weights = tf.constant(weights)
        def weighted_loss_func(y_true, y_pred):
            weight_vec = tf.linalg.matvec(y_true, loss_weights)
            return weight_vec * loss_func(y_true, y_pred)
        return weighted_loss_func
    inputs, output_layer = get_model(
        token_num=50000,
        head_num=5,
        transformer_num=num_transformers,
        embed_dim=100,
        feed_forward_dim=100,
        seq_len=150,
        pos_num=150,
        dropout_rate=0.05,
        training=False,
        trainable=True,
        output_layer_num=1
    )

    extract_layer = Extract(index=0, name='Extract')(output_layer)
    feed_forward_1 = Dense(units=100, name="feed_forward_1")(extract_layer)
    output_logits = Dense(
        units=5,
        activation='softmax',
        name='NSP',
    )(feed_forward_1)

    model = Model(inputs, [output_logits])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=weighted_loss(sparse_categorical_crossentropy), metrics=['accuracy'])
    return model

def train_model(model, train_seqs, train_labels, num_epochs, save_as, batch_size=64, validation_split=.2):
    save_file = os.path.join(models_dir, save_as)
    checkpoint_file = os.path.join(checkpoint_dir, f"{save_as}.ckpt")
    cp_callback = ModelCheckpoint(filepath=checkpoint_file, verbose=1)
    training_result = model.fit(train_seqs, train_labels, epochs=num_epochs, batch_size=batch_size, validation_split=.2, callbacks=[cp_callback])
    model.save(save_file)


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

    def predict(self, preprocessed_inputs):
        return self.keras_model.predict(preprocessed_inputs)

class EnsembleModel(YelpModel):
    def __init__(self, config):
        self.num_models = len(config)
        self.models = [model for model, _ in config]
        self.weights = [weight for _, weight in config]
            
    # averages the softmax probabilites
    def predict_ratings(self, preprocessed_inputs):
        assert len(preprocessed_inputs) == self.num_models
        num_samples = len(preprocessed_inputs[0])
        if num_samples == 2:   # dumbass hard code to fix char inputs - np.ma.size(..., axis=-2) didn't work
            num_samples = len(preprocessed_inputs[0][0])
        predictions = np.zeros((num_samples, 5))
        for i, inputs in enumerate(preprocessed_inputs):
            predictions += self.weights[i] * self.models[i].predict(inputs)
        return [np.argmax(p) + 1 for p in predictions]

    def all_probs(self, preprocessed_inputs):
        return np.array([self.models[i].predict(pi) for i, pi in enumerate(preprocessed_inputs)])

    def copy(self):
        clone = EnsembleModel([])
        clone.models = self.models
        clone.weights = self.weights.copy()
        clone.num_models = self.num_models
        return clone
      
def load_keras_model(name, custom_objects={}, compile=True):
    return load_model(os.path.join(models_dir, name), custom_objects=custom_objects, compile=compile)

def load_custom_model(name, loss_func, custom_objects={}, metrics=[]):
    model = load_keras_model(name, custom_objects=custom_objects, compile=False)
    model.compile(optimizer=Adam(), loss=loss_func, metrics=metrics)
    return model


####### MODELS ########

weighted_star_loss = weighted_loss(star_squared_error)
glove_gru_bi = YelpModel(load_keras_model("glove_gru_bi"))
glove_gru_bi_char = YelpModel(load_keras_model("glove_gru_bi_char"))
gru_bi_50000 = YelpModel(load_keras_model("gru_bi_50000"))
gru_bi_50000_star_loss = YelpModel(load_custom_model("gru_bi_50000_star_loss", star_squared_error, metrics=['sparse_categorical_accuracy']))
gru_bi_50000_wsl = YelpModel(load_custom_model("gru_bi_50000_wsl", weighted_star_loss, metrics=['sparse_categorical_accuracy']))
gru_bi_char = YelpModel(load_keras_model("gru_bi_char"))
gru_bi_char_wscc = YelpModel(load_custom_model("gru_bi_char_wscc", weighted_loss(sparse_categorical_crossentropy)))
bert_model = YelpModel(load_transformer("bert_model_6_wscc_epoch_11.h5"))
BIG_ENSEMBLE = EnsembleModel([
                          (glove_gru_bi, 0.05), (glove_gru_bi_char, 0.15),
                          (gru_bi_50000, 0.2), (gru_bi_50000_star_loss, 0.), (gru_bi_50000_wsl, 0.25),
                          (gru_bi_char, 0.2), (gru_bi_char_wscc, 0.15),
                          (bert_model, 0.)])

#######################

BEST_MODEL = BIG_ENSEMBLE

"""
{'acc': [0.5575221238938053, [0.05, 0.1, 0.05, 0.05, 0.4, 0.1, 0.25, 0.0]], 'err': [0.8254670599803343, [0.0, 0.05, 0.2, 0.15, 0.55, 0.0, 0.0, 0.05]], 'score': [-0.272369714847591, [0.0, 0.0, 0.0, 0.15, 0.15, 0.15, 0.4, 0.15]]}
{'acc': [0.5932203389830508, [0.05, 0.15, 0.2, 0.0, 0.25, 0.2, 0.15, 0.0]], 'err': [0.9015645371577575, [0.0, 0.0, 0.3, 0.15, 0.35, 0.2, 0.0, 0.0]], 'score': [-0.31551499348109513, [0.05, 0.1, 0.2, 0.05, 0.25, 0.2, 0.15, 0.0]]}
{'acc': [0.6595744680851063, [0.0, 0.45, 0.1, 0.1, 0.15, 0.0, 0.2, 0.0]], 'err': [0.3916827852998066, [0.2, 0.05, 0.0, 0.1, 0.05, 0.05, 0.5, 0.05]], 'score': [0.2659574468085106, [0.1, 0.3, 0.05, 0.05, 0.0, 0.15, 0.3, 0.05]]}
╒═══════════════════════════════════════════╤═════════════════════════╤═════════════════════════╤═════════════════════════╤═════════════════════════╤═════════════════════════╕
│ Model                                     │ OVERALL                 │ CHALLENGE 3             │ CHALLENGE 5             │ CHALLENGE 6             │ CHALLENGE 8             │
│                                           │ star error | accuracy   │ star error | accuracy   │ star error | accuracy   │ star error | accuracy   │ star error | accuracy   │
╞═══════════════════════════════════════════╪═════════════════════════╪═════════════════════════╪═════════════════════════╪═════════════════════════╪═════════════════════════╡
│ big_ensemble_results_ACC                  │ 0.838 *     55.605      │ 0.380       64.419 *    │ 0.532 *     51.800 *    │ 2.002       41.000      │ 0.436       65.200      │
├───────────────────────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┤
│ big_ensemble_results_ERR                  │ 0.833       55.158 *    │ 0.390       64.232      │ 0.552       49.200      │ 1.958       41.800      │ 0.432       65.400      │
├───────────────────────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┤
│ big_ensemble_results_SCORE                │ 0.835       55.446      │ 0.408       61.985      │ 0.556       50.200      │ 1.950       42.800      │ 0.426       66.800      │
├───────────────────────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┤
│ big_ensemble_no_challenge_5_results_ACC   │ 0.847       55.245      │ 0.384 *     64.981      │ 0.620       43.400      │ 1.974       45.600      │ 0.412       67.000      │
├───────────────────────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┤
│ big_ensemble_no_challenge_5_results_ERR   │ 0.846       54.390      │ 0.421       62.360      │ 0.646       42.600      │ 1.908 *     46.000 *    │ 0.408       66.600      │
├───────────────────────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┤
│ big_ensemble_no_challenge_5_results_SCORE │ 0.843       55.111      │ 0.393       64.045      │ 0.618       43.600      │ 1.958       45.600      │ 0.402       67.200 *    │
├───────────────────────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┤
│ big_ensemble_challenges_3_8_results_ACC   │ 0.894       53.402      │ 0.390       64.607      │ 0.684       37.600      │ 2.092       44.000      │ 0.410 *     67.400      │
├───────────────────────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┤
│ big_ensemble_challenges_3_8_results_ERR   │ 0.859       54.668      │ 0.382       63.670      │ 0.588       46.400      │ 2.062       41.200      │ 0.402 *     67.400 *    │
├───────────────────────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┤
│ big_ensemble_challenges_3_8_results_SCORE │ 0.883       54.049      │ 0.380       64.794 *    │ 0.650       40.800      │ 2.094       43.400      │ 0.408       67.200      │
╘═══════════════════════════════════════════╧═════════════════════════╧═════════════════════════╧═════════════════════════╧═════════════════════════╧═════════════════════════╛
"""
