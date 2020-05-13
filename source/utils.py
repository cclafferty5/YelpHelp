import tensorflow as tf
import sys
from collections import Counter
import numpy as np
import json
import os
from tabulate import tabulate

from keras.preprocessing.text import Tokenizer, tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, Dense, Dropout, GRU, Bidirectional
from keras.callbacks import ModelCheckpoint
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam

cwd = os.getcwd()
root_dir = os.path.join(cwd, "..")
models_dir = os.path.join(cwd, "models")
checkpoint_dir = os.path.join(models_dir, "checkpoints")
dataset_dir = os.path.join(cwd, "../datasets")
dataset_name = "yelp_review_training_dataset.jsonl"
tokenizers_dir = os.path.join(models_dir, "tokenizers")
test_set_dir = os.path.join(root_dir, "test-sets")
ensemble_dir = os.path.join(root_dir, "ensembles")


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


def get_texts_and_labels(dataset):
    texts = [d["text"] for d in dataset]
    labels = [d["stars"] - 1 for d in dataset]
    return texts, labels

def load_tokenizer(name):
    file_path = os.path.join(tokenizers_dir, name)
    with open(file_path) as tkf:
        return tokenizer_from_json(tkf.read())

def train_model(model, train_seqs, train_labels, num_epochs, save_as, batch_size=64, validation_split=.2, save_weights=False):
    save_file = os.path.join(models_dir, save_as)
    checkpoint_file = os.path.join(checkpoint_dir, f"{save_as}.ckpt")
    cp_callback = ModelCheckpoint(filepath=checkpoint_file, verbose=1, save_weights_only=save_weights)
    training_result = model.fit(train_seqs, train_labels, epochs=num_epochs, batch_size=batch_size, validation_split=.2, callbacks=[cp_callback])
    model.save(save_file)

def predict_on_texts(model, texts, preprocessor, actual_stars=None):
    inputs = preprocessor.preprocess(texts)
    predictions = model.predict(inputs)
    for i, p in enumerate(predictions):
        print("---------------------")
        print("TEXT:\n{}\nPREDICTED STARS:{}".format(texts[i], np.argmax(p) + 1))
        if actual_stars:
            print("ACTUAL STARS: {}".format(actual_stars[i]))

def get_balanced_dataset(dataset, size=1000):
    class_counter = Counter()
    result = []
    ration = size // 5
    finished = set()
    for d in dataset:
        star = d["stars"]
        if star not in finished:
            class_counter[star] += 1
            result.append(d)
            if class_counter[star] >= ration:
                finished.add(star)
        if len(finished) == 5:
            return result
    return result
        

def predict_from_data(model, dataset, preprocessor):
    stars = None
    if "stars" in dataset[0]:
        stars = [d["stars"] for d in dataset]
    texts = [d["text"] for d in dataset]
    predict_on_texts(model, texts, preprocessor, actual_stars=stars)

def batch_predict(batch, model, preprocessor):
    texts = [b["text"] for b in batch]
    batch_input = preprocessor.preprocess(texts)
    predictions = model.predict_ratings(batch_input)
    assert len(batch) == len(predictions)
    for i, b in enumerate(batch):
        b["predicted_stars"] = predictions[i]
        
def predict_test_set(test_set, model, preprocessor, batch_size=64, show_accuracy=True, print_results=True):
    for i in range(0, len(test_set), batch_size):
        batch = test_set[i: i + batch_size]
        batch_predict(batch, model, preprocessor)
    accuracy, avg_star_error = None, None
    if show_accuracy:
        accuracy = (len([d for d in test_set if d["stars"] == d["predicted_stars"]]) / len(test_set)) * 100
        avg_star_error = sum([abs(d["predicted_stars"] - d["stars"]) for d in test_set]) / len(test_set)
        if print_results:
            print("Accuracy: {:.3f}".format(accuracy))
            print("Average Star Error: {:.5f}".format(avg_star_error))
    return accuracy, avg_star_error

def load_data_set(name, test_set=False):
    set_dir = test_set_dir if test_set else dataset_dir
    with open(os.path.join(set_dir, name)) as df:
        return [json.loads(line) for line in df]

def load_keras_model(name, custom_objects={}, compile=True):
    return load_model(os.path.join(models_dir, name), custom_objects=custom_objects, compile=compile)

def load_custom_model(name, loss_func, custom_objects={}, metrics=[]):
    model = load_keras_model(name, custom_objects=custom_objects, compile=False)
    model.compile(optimizer=Adam(), loss=loss_func, metrics=metrics)
    return model

def load_transformer(name):
    weights_file = os.path.join(models_dir, name)
    model = build_transformer_model()
    model.load_weights(weights_file)
    return model

def compare_class_accuracies(test_set, models_and_preprocs):
    results = {}
    def class_result(c):
        relevant = [d for d in test_set if int(d["stars"]) == c]
        acc = len([d for d in relevant if d["stars"] == d["predicted_stars"]]) / len(relevant)
        star_err = sum([abs(d["stars"] - d["predicted_stars"]) for d in relevant]) / len(relevant)
        return acc, star_err
    for name, model_and_preproc in models_and_preprocs.items():
        model, preprocessor = model_and_preproc
        avg_acc, avg_se = predict_test_set(test_set, model, preprocessor, print_results=False)
        result = results[name] = [(avg_acc, avg_se)]
        result += [class_result(c) for c in range(1, 6)]
    headers = ["Model", "OVERALL\naccuracy | star error"]
    headers += ["{}\naccuracy | star error".format(star) for star in range(1, 6)]
    def format_result(result):
        acc, star_error = result
        return "{:.3f}     {:.3f}".format(acc, star_error)
    table = [[name] + [format_result(r) for r in result] for name, result in results.items()]
    print(tabulate(table, headers, tablefmt='fancy_grid'))

def compare_on_test_sets(test_sets, models_and_preprocs, show_results=True):
    results = {name: {} for name in models_and_preprocs}
    for model_name, model_and_preproc in models_and_preprocs.items():
        result = results[model_name]
        overall = result['overall'] = [0, 0]
        for test_name, test_set in test_sets.items():
            model, preprocessor = model_and_preproc
            acc, star_error = predict_test_set(test_set, model, preprocessor, print_results=False)
            result[test_name] = [acc, star_error]
            overall[0] += acc
            overall[1] += star_error
        overall[0] /= len(test_sets)
        overall[1] /= len(test_sets)
    table = []
    test_names = list(test_sets.keys())
    headers = ["Model", "OVERALL\nstar error | accuracy"]
    headers += ["{}\nstar error | accuracy".format(name) for name in test_names]
    def format_result(result, col_name):
        acc, star_error = result
        is_best_acc = acc == max([r[col_name][0] for r in results.values()])
        is_best_star_error = star_error == min([r[col_name][1] for r in results.values()])
        return "{:.3f}{}     {:.3f}{}".format(star_error, " *" if is_best_acc else "  ", acc, " *" if is_best_star_error else "  ")
    for model_name, result in results.items():
        row = [model_name, format_result(result['overall'], 'overall')]
        row += [format_result(result[name], name) for name in test_names]
        table.append(row)
    if show_results:
        print(tabulate(table, headers, tablefmt='fancy_grid'))
    return results

    
def generate_weights(num_models, depth=.05):
    divisor = int(1 / depth)
    def generate_helper(current_weights, left):
        if sum(current_weights) > divisor:
            return
        elif left <= 1:
            for w in range(21):
                if w + sum(current_weights) == divisor:
                    yield [weight / divisor for weight in current_weights + [w]]
        else:
            for w in range(21):
                yield from generate_helper(current_weights + [w], left - 1)
    yield from generate_helper([], num_models)

def best_weights_given_probs(probs, labels):
    bests = {'acc': [0, []], 'err': [5, []], 'score': [-100, []]}
    num_samples = len(probs[0])
    for weights in generate_weights(len(probs)):
        average_probs = np.average(probs, axis=0, weights=weights)
        predictions = np.argmax(average_probs, axis=1)
        acc = np.sum(predictions == labels) / num_samples
        star_err = np.sum(np.abs(predictions - labels)) / num_samples
        score = acc - star_err
        if acc > bests['acc'][0]:
            bests['acc'] = [acc, weights]
        if star_err < bests['err'][0]:
            bests['err'] = [star_err, weights]
        if score > bests['score'][0]:
            bests['score'] = [score, weights]
    return bests

def best_weights(ensemble_model, ensemble_preproc, test_set):
    texts, labels = get_texts_and_labels(test_set)
    inputs = ensemble_preproc.preprocess(texts)
    probs = ensemble_model.all_probs(inputs)
    return best_weights_given_probs(probs, labels)

def save_bests(bests, save_name):
    with open(os.path.join(ensemble_dir, save_name), "w+") as snf:
        print(json.dumps(bests), file=snf)

def load_bests(save_name):
    with open(os.path.join(ensemble_dir, save_name)) as snf:
        return json.load(snf)

def get_mps_for_bests(ensemble, preproc, names):
    models_and_preprocs = {}
    bests = [load_bests(name) for name in names]
    for name, best in zip(names, bests):
        for met in ('ACC', 'ERR', 'SCORE'):
            cur_ens = ensemble.copy()
            cur_ens.weights = best[met.lower()][1]
            models_and_preprocs[f"{name}_{met}"] = (cur_ens, preproc)
    return models_and_preprocs
