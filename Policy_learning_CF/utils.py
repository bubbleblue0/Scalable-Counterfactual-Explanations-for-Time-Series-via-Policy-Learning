import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sktime.datasets import load_from_tsfile
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
from scipy.spatial import distance
from tslearn.utils import to_sklearn_dataset
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import csv
from datetime import datetime

import os
import argparse
from typing import Dict
from utils import *
# Core dependencies
try:
    import tensorflow as tf
    import tensorflow.keras as keras
except ImportError:
    print("TensorFlow not found. Please install: pip install tensorflow")
    exit(1)

try:
    import wandb
except ImportError:
    print("wandb not found. Please install: pip install wandb")
    exit(1)

NUM_SAMPLES = 5  # For visualization
DEFAULT_OUTPUT_DIR = "rl_32_ela_kdd_rebuttal_results"

# python
def z_score_normalize(X):
    # supports 2D (n, L) and 3D (n, C, L)
    if X.ndim == 2:
        mean = np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, keepdims=True) + 1e-8
        return (X - mean) / std
    if X.ndim == 3:
        mean = np.mean(X, axis=2, keepdims=True)   # mean over time axis
        std = np.std(X, axis=2, keepdims=True) + 1e-8
        return (X - mean) / std
    raise ValueError("Unsupported input ndim")


def label_encoder(training_labels, testing_labels):
    le = preprocessing.LabelEncoder()
    le.fit(np.concatenate((training_labels, testing_labels), axis=0))
    y_train = le.transform(training_labels)
    y_test = le.transform(testing_labels)

    return y_train, y_test

def readUCR(ds_name):
    path = "/UCRArchive_2018/"
    train_data = np.loadtxt(path + ds_name +'/' + ds_name+ '_TRAIN.tsv' , delimiter='\t')
    x_train = train_data[:, 1:]
    y_train = train_data[:, 0]
    # print(x_train.shape, y_train.shape)

    test_data = np.loadtxt(path + ds_name + '/' + ds_name + '_TEST.tsv', delimiter='\t')
    x_test = test_data[:, 1:]
    y_test = test_data[:, 0]

    y_train, y_test = label_encoder(y_train, y_test)
    # print(x_test.shape, y_test.shape)


    print("Applying z-score normalization for dataset ", ds_name)
    x_train = z_score_normalize(x_train)
    x_test = z_score_normalize(x_test)


    return x_train, y_train, x_test, y_test

def readUEA(name):
    DATA_PATH = "/MTS_DATA"

    X_train, y_train = load_from_tsfile(os.path.join(DATA_PATH, name, name + '_TRAIN.ts'),
                                        return_data_type='numpy3d')
    X_test, y_test = load_from_tsfile(os.path.join(DATA_PATH, name, name + '_TEST.ts'),
                                      return_data_type='numpy3d')
    # return X_train, X_test, y_train, y_test  # return raw labels
    if name in ["BasicMotions","RacketSports"]:
        print("Applying z-score normalization for dataset ", name)
        # Z-score normalization
        X_train = z_score_normalize(X_train)
        X_test = z_score_normalize(X_test)

    y_train, y_test = label_encoder(y_train, y_test)
    # print(x_test.shape, y_test.shape)
    return X_train, y_train, X_test, y_test

def getmetrics(x1, x2):
    x1 = np.round(x1, 3)
    x2 = np.round(x2, 3)

    l = np.round(x1 - x2, 3)
    l1 = distance.cityblock(x1, x2)
    l2 = np.linalg.norm(x1 - x2)  # Correct usage of np.linalg.norm for one-dimensional arrays
    l_inf = distance.chebyshev(x1, x2)
    sparsity = (len(l) - np.count_nonzero(l)) / len(l)

    segnums = get_segmentsNumber(l)
    return l1, l2, l_inf, sparsity, segnums

def get_segmentsNumber(l4):
    flag, count = 0,0
    for i in range(len(l4)):
        if l4[i:i+1][0]!=0:
            flag=1
        if flag==1 and l4[i:i+1][0]==0:
            count= count+1
            flag=0
    return count

def cf_ood(X_train, counterfactual_examples):

    # Local Outlier Factor (LOF)
    lof = LocalOutlierFactor(n_neighbors=int(np.sqrt(len(X_train))), novelty=True, metric='euclidean')
    lof.fit(to_sklearn_dataset(X_train))

    novelty_detection = lof.predict(to_sklearn_dataset(counterfactual_examples))

    ood= np.count_nonzero(novelty_detection == -1)
    OOD_lof = ood / len(counterfactual_examples)

    # One-Class SVM (OC-SVM)
    clf = OneClassSVM(gamma='scale', nu=0.02).fit(to_sklearn_dataset(X_train))

    novelty_detection = clf.predict(to_sklearn_dataset(counterfactual_examples))

    ood = np.count_nonzero(novelty_detection == -1)
    OOD_svm = ood/ len(counterfactual_examples)

    # Initialize a list to store OOD results for min_edit_cf
    OOD_ifo = []

    # Loop over different random seeds
    for seed in range(10):
        iforest = IsolationForest(random_state=seed).fit(to_sklearn_dataset(X_train))

        novelty_detection = iforest.predict(to_sklearn_dataset(counterfactual_examples))

        ood = np.count_nonzero(novelty_detection == -1)

        OOD_ifo.append((ood/ len(counterfactual_examples)))

    mean_OOD_ifo = np.mean(OOD_ifo)

    return OOD_svm, OOD_lof, mean_OOD_ifo

def prepare_data(X_train, Y_train, X_test, Y_test):
    """Prepare and encode training/test data."""
    # Expand dimensions for Conv1D layers
    X_train = np.expand_dims(X_train, axis=-1).astype(np.float32)
    X_test = np.expand_dims(X_test, axis=-1).astype(np.float32)

    # Encode labels
    label_encoder = LabelEncoder()
    Y_all = np.concatenate((Y_train, Y_test), axis=0)
    label_encoder.fit(Y_all)
    Y_train_int = label_encoder.transform(Y_train)
    Y_test_int = label_encoder.transform(Y_test)

    # One-hot encode
    onehot_encoder = OneHotEncoder(sparse_output=False, categories='auto')
    Y_train = onehot_encoder.fit_transform(Y_train_int.reshape(-1, 1))
    Y_test = onehot_encoder.transform(Y_test_int.reshape(-1, 1))

    return X_train, Y_train, X_test, Y_test, onehot_encoder

def configure_gpu():
    """Configure TensorFlow GPU memory growth."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")

def save_metrics_to_csv(ds_name, metrics_dict):
    """Save training metrics to CSV file."""

    csv_path = os.path.join(DEFAULT_OUTPUT_DIR, "results", f"{ds_name}_metrics.csv")
    os.makedirs(os.path.join(DEFAULT_OUTPUT_DIR, "results"), exist_ok=True)

    # Add timestamp
    metrics_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Write to CSV
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = metrics_dict.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics_dict)

    print(f"Metrics saved to: {csv_path}")

def create_datasets(X_train, Y_train, X_test, Y_test, BATCH_SIZE):
    """Create TensorFlow datasets for training."""
    # Classifier datasets
    BUFFER_SIZE = min(1024, X_train.shape[0])

    trainset_classifier = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    trainset_classifier = trainset_classifier.shuffle(buffer_size=BUFFER_SIZE).batch(BATCH_SIZE)

    testset_classifier = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    testset_classifier = testset_classifier.shuffle(buffer_size=BUFFER_SIZE).batch(BATCH_SIZE)

    # Autoencoder datasets
    trainset_ae = tf.data.Dataset.from_tensor_slices(X_train)
    trainset_ae = trainset_ae.map(lambda x: (x, x))
    trainset_ae = trainset_ae.shuffle(buffer_size=BUFFER_SIZE).batch(BATCH_SIZE)

    testset_ae = tf.data.Dataset.from_tensor_slices(X_test)
    testset_ae = testset_ae.map(lambda x: (x, x))
    testset_ae = testset_ae.shuffle(buffer_size=BUFFER_SIZE).batch(BATCH_SIZE)

    return trainset_classifier, testset_classifier, trainset_ae, testset_ae


def generate_targets(X, predictor, num_classes, onehot_encoder):
    """Generate counterfactual targets for given data."""
    Y_probs = predictor(X)
    Y_m = np.argmax(Y_probs, axis=1)

    Y_t_int = []
    for i, (probs, orig_class) in enumerate(zip(Y_probs, Y_m)):
        sorted_classes = np.argsort(probs)[::-1]

        # Find first class different from original
        for class_idx in sorted_classes:
            if class_idx != orig_class:
                Y_t_int.append(class_idx)
                break
        else:
            # Fallback: random different class
            available_classes = [c for c in range(num_classes) if c != orig_class]
            Y_t_int.append(np.random.choice(available_classes))

    Y_t_int = np.array(Y_t_int)
    Y_t = onehot_encoder.transform(Y_t_int.reshape(-1, 1))

    return Y_t, Y_t_int, Y_m

def calculate_embedding_separation(ae, X_test, Y_test):
    """Calculate separation between classes in embedding space."""
    embeddings = ae.encoder(X_test).numpy()
    labels = np.argmax(Y_test, axis=1)

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    # Calculate mean embeddings per class
    class_means = []
    for label in unique_labels:
        class_mask = labels == label
        class_embeddings = embeddings[class_mask]
        class_means.append(np.mean(class_embeddings, axis=0))

    # Calculate average pairwise distance between class means
    distances = []
    for i in range(len(class_means)):
        for j in range(i + 1, len(class_means)):
            dist = np.linalg.norm(class_means[i] - class_means[j])
            distances.append(dist)

    return np.mean(distances) if distances else 0.0

def evaluate_and_visualize_results(explanation, Y_t_test, ds_name, predictor) -> Dict[str, float]:
    """Evaluate and visualize final results."""
    # Get actual classifier predictions
    orig_predictions = predictor(explanation.data['orig']['X'])
    cf_predictions = predictor(explanation.data['cf']['X'])

    # Convert to class labels
    orig_labels = np.argmax(orig_predictions, axis=1)
    cf_labels = np.argmax(cf_predictions, axis=1)
    target_labels = np.argmax(Y_t_test, axis=1)

    # Print evaluation metrics
    print("=== FINAL VERIFICATION ===")
    print(f"Original labels: {orig_labels[:NUM_SAMPLES]}")
    print(f"CF labels: {cf_labels[:NUM_SAMPLES]}")
    print(f"Target labels: {target_labels[:NUM_SAMPLES]}")
    print(f"Total samples in test set: {len(cf_labels)}")
    print(f"Success rate: {np.mean(cf_labels == target_labels):.2%}")
    print(f"Flip rate: {np.mean(cf_labels != orig_labels):.2%}")

    # Create visualization
    fig, ax = plt.subplots(2, NUM_SAMPLES, figsize=(25, 10))

    for i in range(NUM_SAMPLES):
        ax[0][i].plot(explanation.data['orig']['X'][i])
        ax[1][i].plot(explanation.data['cf']['X'][i])

        ax[0][i].set_xlabel(f"Label: {orig_labels[i]}")
        ax[1][i].set_xlabel(f"Label: {cf_labels[i]}")

    ax[0][0].set_ylabel("X")
    ax[1][0].set_ylabel("X_hat")

    # Save and log

    os.makedirs(f'{DEFAULT_OUTPUT_DIR}/figs', exist_ok=True)
    fig.savefig(os.path.join(f'{DEFAULT_OUTPUT_DIR}/figs', f'final_plot_{ds_name}.png'))
    wandb.log({"Final_Results": wandb.Image(fig)})
    plt.show()

def evaluate_and_get_metrics(explanation, Y_t_test, ds_name, predictor) -> Dict[str, float]:
    """Evaluate and return final metrics."""
    # Get actual classifier predictions
    orig_predictions = predictor(explanation.data['orig']['X'])
    cf_predictions = predictor(explanation.data['cf']['X'])

    # Convert to class labels
    orig_labels = np.argmax(orig_predictions, axis=1)
    cf_labels = np.argmax(cf_predictions, axis=1)
    target_labels = np.argmax(Y_t_test, axis=1)

    # Calculate metrics
    success_rate = np.mean(cf_labels == target_labels)
    flip_rate = np.mean(cf_labels != orig_labels)

    # Print evaluation metrics
    print("=== FINAL VERIFICATION ===")
    print(f"Original labels: {orig_labels[:NUM_SAMPLES]}")
    print(f"CF labels: {cf_labels[:NUM_SAMPLES]}")
    print(f"Target labels: {target_labels[:NUM_SAMPLES]}")
    print(f"Total samples in test set: {len(cf_labels)}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Flip rate: {flip_rate:.2%}")

    # Create visualization
    fig, ax = plt.subplots(2, NUM_SAMPLES, figsize=(25, 10))

    for i in range(NUM_SAMPLES):
        ax[0][i].plot(explanation.data['orig']['X'][i])
        ax[1][i].plot(explanation.data['cf']['X'][i])

        ax[0][i].set_xlabel(f"Label: {orig_labels[i]}")
        ax[1][i].set_xlabel(f"Label: {cf_labels[i]}")

    ax[0][0].set_ylabel("X")
    ax[1][0].set_ylabel("X_hat")

    # Save plot
    os.makedirs(f'{DEFAULT_OUTPUT_DIR}/figs', exist_ok=True)
    fig.savefig(os.path.join(f'{DEFAULT_OUTPUT_DIR}/figs', f'final_plot_{ds_name}.png'))
    plt.close()

    print(f"Final results plot saved to: {DEFAULT_OUTPUT_DIR}/figs/final_plot_{ds_name}.png")

    return {
        'success_rate': success_rate,
        'flip_rate': flip_rate,
        'total_test_samples': len(cf_labels)
    }
def visualize_autoencoder_reconstruction(ae, X_test, ds_name):
    """Visualize autoencoder reconstruction quality."""
    np.random.seed(0)
    indices = np.random.choice(X_test.shape[0], NUM_SAMPLES, replace=False)
    x_samples = X_test[indices]
    x_recon = ae.predict(x_samples)

    plt.figure(figsize=(20, 6))
    for i in range(NUM_SAMPLES):
        plt.subplot(2, NUM_SAMPLES, i + 1)
        plt.plot(x_samples[i].squeeze(), label="Original")
        plt.title("Original")
        plt.xticks([]), plt.yticks([])

        plt.subplot(2, NUM_SAMPLES, i + 1 + NUM_SAMPLES)
        plt.plot(x_recon[i].squeeze(), label="Reconstruction")
        plt.title("Reconstruction")
        plt.xticks([]), plt.yticks([])

    plt.tight_layout()

    # Save plot only
    os.makedirs(f"{DEFAULT_OUTPUT_DIR}/figs", exist_ok=True)
    ae_recon_path = os.path.join(f"{DEFAULT_OUTPUT_DIR}/figs", f"ae_reconstruction_{ds_name}.png")
    plt.savefig(ae_recon_path)
    plt.close()

    print(f"Autoencoder reconstruction plot saved to: {ae_recon_path}")

