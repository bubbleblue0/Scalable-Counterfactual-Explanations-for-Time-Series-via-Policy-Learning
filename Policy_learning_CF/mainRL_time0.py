#!/usr/bin/env python3
"""
Time Series Counterfactual Explanation with Reinforcement Learning
Main script for training and evaluating CFRL on UCR time series datasets.

Usage: python mainRL_time0.py --dataset <dataset_name>
Environment: tf_gpu
"""
import csv
from datetime import datetime
import time

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
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

from alibi.explainers import CounterfactualRL
from alibi.explainers.cfrl_base import Callback
from alibi.models.tensorflow.cfrl_models import TimeEncoder, TimeDecoder, TimeSeriesClassifier
import utils  # added to allow overriding utils.DEFAULT_OUTPUT_DIR at runtime

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

# Training constants
BATCH_SIZE = 128
LATENT_DIM = 32 # 16 for Handoutlines, 64 for FordA and FordB, res still saved into rl_32_ela_final, the others should be 32
# LATENT_DIM = 64 # for FordA and FordB
EPOCHS_AE = 1500
BATCH_SIZE_AE = 128


# CFRL constants
COEFF_SPARSITY = 1
COEFF_CONSISTENCY = 0
TRAIN_STEPS = 50000

# Wandb configuration
WANDB_PROJECT = "RL-CF-final-50000-1-32-ela-binary-kdd-rebuttal"
DEFAULT_OUTPUT_DIR = "rl_32_ela_kdd_rebuttal_results_with_replay"

# =============================================================================
# MODEL CLASSES
# =============================================================================

class AE(keras.Model):
    """Autoencoder model wrapper."""

    def __init__(self, encoder: keras.Model, decoder: keras.Model, **kwargs) -> None:
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, x: tf.Tensor, **kwargs):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


# =============================================================================
# CALLBACK CLASSES
# =============================================================================

class RewardCallback(Callback):
    """Callback to log reward during training."""

    def __call__(self, step: int, update: int, model: CounterfactualRL,
                 sample: Dict[str, np.ndarray], losses: Dict[str, float]):
        if step % 100 != 0:
            return

        X_cf = sample["X_cf"]
        Y_t = sample["Y_t"]
        Y_m_cf = model.params["predictor"](X_cf)

        reward = np.mean(model.params["reward_func"](Y_m_cf, Y_t))
        wandb.log({"reward": reward, "step": step})


class SuccessRateCallback(Callback):
    """Callback to log success and flip rates."""

    def __call__(self, step, update, model, sample, losses):
        if step % 100 != 0:
            return

        Y_m = np.argmax(sample["Y_m"], axis=1)
        Y_t = np.argmax(sample["Y_t"], axis=1)
        Y_m_cf = np.argmax(model.params["predictor"](sample["X_cf"]), axis=1)

        success_rate = np.mean(Y_m_cf == Y_t)
        flip_rate = np.mean(Y_m_cf != Y_m)
        targets_different = np.mean(Y_m != Y_t)

        wandb.log({
            "success_rate": success_rate,
            "flip_rate": flip_rate,
            "targets_different": targets_different,
            "step": step
        })


class ImprovedTimeSeriesCallback(Callback):
    """Callback for visualizing time series counterfactuals."""

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, step: int, update: int, model: CounterfactualRL,
                 sample: Dict[str, np.ndarray], losses: Dict[str, float]):
        if step % 100 != 0:
            return

        predictor = model.params["predictor"]
        X = sample["X"][:NUM_SAMPLES]
        X_cf = sample["X_cf"][:NUM_SAMPLES]
        diff = np.abs(X - X_cf)

        # Get predictions
        Y_m = np.argmax(sample["Y_m"][:NUM_SAMPLES], axis=1).astype(int)

        # Generate improved targets
        Y_probs = predictor(X)
        Y_t_improved = []
        for i, (probs, orig_class) in enumerate(zip(Y_probs, Y_m)):
            sorted_classes = np.argsort(probs)[::-1]
            for class_idx in sorted_classes:
                if class_idx != orig_class:
                    Y_t_improved.append(class_idx)
                    break
            else:
                available_classes = [c for c in range(self.num_classes) if c != orig_class]
                Y_t_improved.append(np.random.choice(available_classes))

        Y_t = np.array(Y_t_improved)
        Y_m_cf = np.argmax(predictor(X_cf), axis=1).astype(int)

        print(f"Step {step}: Original: {Y_m}, Target: {Y_t}, CF: {Y_m_cf}")

        # Create visualization
        fig, axes = plt.subplots(3, NUM_SAMPLES, figsize=(25, 10))
        for i in range(NUM_SAMPLES):
            axes[0, i].plot(X[i].squeeze())
            axes[0, i].set_title(f"Input: {Y_m[i]}")

            axes[1, i].plot(X_cf[i].squeeze())
            axes[1, i].set_title(f"CF: {Y_m_cf[i]} (Target: {Y_t[i]})")

            axes[2, i].plot(diff[i].squeeze())
            axes[2, i].set_title("Difference")

        plt.tight_layout()
        wandb.log({"samples": wandb.Image(fig), "step": step})
        plt.close(fig)


class LossCallback(Callback):
    """Callback to log training losses."""

    def __call__(self, step: int, update: int, model: CounterfactualRL,
                 sample: Dict[str, np.ndarray], losses: Dict[str, float]):
        if (step + update) % 100 == 0:
            wandb.log({**losses, "step": step})


# =============================================================================
# MODEL TRAINING FUNCTIONS
# =============================================================================

def train_or_load_classifier(X_train, Y_train, X_test, Y_test, num_classes, ds_name):
    """Train or load time series classifier."""
    classifier_path = os.path.join("tensorflow_32", f"{ds_name}_Time_Classifier")
    os.makedirs(classifier_path, exist_ok=True)

    if len(os.listdir(classifier_path)) == 0:
        # Train new classifier
        classifier = TimeSeriesClassifier(output_dim=num_classes)

        optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        loss = keras.losses.CategoricalCrossentropy(from_logits=False)
        classifier.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=100, restore_best_weights=True
        )
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=30, min_lr=1e-5
        )

        # Training parameters
        batch_size = 16
        nb_epochs = 2000
        mini_batch_size = int(min(X_train.shape[0] / 10, batch_size))

        # Train classifier
        classifier.fit(
            X_train, Y_train,
            validation_data=(X_test, Y_test),
            epochs=nb_epochs,
            batch_size=mini_batch_size,
            callbacks=[reduce_lr, early_stop],
            verbose=True
        )
        classifier.save(classifier_path)
    else:
        # Load existing classifier
        classifier = keras.models.load_model(classifier_path)

    # Evaluate classifier
    loss, accuracy = classifier.evaluate(X_test, Y_test)
    print(f"Classifier - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    return classifier


def train_or_load_autoencoder_with_metrics(trainset_ae, testset_ae, nb_timesteps, nb_features, ds_name):
    """Train or load autoencoder with metrics tracking."""
    ae_path = os.path.join("tensorflow_32", f"{ds_name}_Time_autoencoder")
    os.makedirs(ae_path, exist_ok=True)

    metrics = {}

    if len(os.listdir(ae_path)) == 0:
        print("Training new autoencoder...")

        # Train new autoencoder
        ae = AE(
            encoder=TimeEncoder(latent_dim=LATENT_DIM),
            decoder=TimeDecoder(n_timesteps=nb_timesteps, n_features=nb_features)
        )

        optimizer = keras.optimizers.Adam(learning_rate=0.005)
        loss = keras.losses.MeanSquaredError()
        ae.compile(optimizer=optimizer, loss=loss)

        # Callbacks
        early_stop_ae = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=50, restore_best_weights=True
        )
        reduce_lr_ae = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=30, min_lr=1e-6
        )

        mini_batch_size_ae = int(min(len(trainset_ae) / 10, BATCH_SIZE_AE))

        # Train autoencoder
        history = ae.fit(
            trainset_ae,
            validation_data=testset_ae,
            epochs=EPOCHS_AE,
            batch_size=mini_batch_size_ae,
            callbacks=[reduce_lr_ae, early_stop_ae],
            verbose=True
        )

        # Save model
        ae.save(ae_path)

        # Extract essential metrics only
        metrics.update({
            'ae_final_val_loss': history.history['val_loss'][-1],
            'ae_best_val_loss': min(history.history['val_loss'])
        })

        # Print and plot results
        print_ae_loss_summary(history.history)
        plot_simple_history(history.history, ds_name)

    else:
        print("Loading existing autoencoder...")
        ae = keras.models.load_model(ae_path)

        # Evaluate on test set
        test_loss = ae.evaluate(testset_ae, verbose=0)
        metrics.update({
            'ae_final_val_loss': test_loss,
            'ae_best_val_loss': test_loss
        })
        print(f"Autoencoder test loss after loading: {test_loss:.6f}")

    return ae, metrics

def print_ae_loss_summary(history_dict):
    """Print simple autoencoder training summary."""
    print("\nAutoencoder Training Summary:")
    print(f"  Epochs trained: {len(history_dict['loss'])}")
    print(f"  Final training loss: {history_dict['loss'][-1]:.6f}")
    print(f"  Final validation loss: {history_dict['val_loss'][-1]:.6f}")
    print(f"  Best validation loss: {min(history_dict['val_loss']):.6f}")


def plot_simple_history(history_dict, ds_name):
    """Create and save simple training history plot."""
    plt.figure(figsize=(10, 4))

    plt.plot(history_dict['loss'], label='Training Loss')
    plt.plot(history_dict['val_loss'], label='Validation Loss')
    plt.title(f'Autoencoder Training History - {ds_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save plot
    os.makedirs(os.path.join(DEFAULT_OUTPUT_DIR, "figs"), exist_ok=True)
    plot_path = os.path.join(DEFAULT_OUTPUT_DIR, "figs", f"ae_history_{ds_name}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Training history plot saved to: {plot_path}")




# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():

    """Main execution function."""
    # Start overall timing
    overall_start_time = time.time()

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run Time Series Counterfactual with Reinforcement Learning")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to use")
    parser.add_argument("--no_replay", action="store_true", default=False,
                        help="Disable experience replay (ablation)")
    args = parser.parse_args()

    # Initialize metrics dictionary
    metrics = {'dataset': args.dataset, 'use_replay': (not args.no_replay)}

    # Set up output directory based on replay mode
    # Create dataset-specific subdirectories to prevent overwriting
    if args.no_replay:
        base_dir = "rl_32_ela_kdd_rebuttal_results_no_replay"
    else:
        base_dir = "rl_32_ela_kdd_rebuttal_results_with_replay"

    global DEFAULT_OUTPUT_DIR
    DEFAULT_OUTPUT_DIR = os.path.join(base_dir, args.dataset)
    utils.DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR

    # Create the directory
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    print(f"Results will be saved to: {DEFAULT_OUTPUT_DIR}")

    # Configure environment
    configure_gpu()

    # Load and prepare data
    print(f"Loading dataset: {args.dataset}")

    # Dataset-specific batch sizes
    DATASET_BATCH_SIZES = {
        "BeetleFly": 20,
        "BirdChicken": 20,
        "Chinatown": 20,
        "Coffee": 28,
        "Computers": 32,
        "ECG200": 16,
        "FordA": 256,
        "FordB": 256,
        "FreezerRegularTrain": 16,
        "GunPoint": 50,
        "GunPointAgeSpan": 16,
        "GunPointMaleVersusFemale": 16,
        "GunPointOldVersusYoung": 16,
        "HandOutlines": 128,
        "Lightning2": 60,
        "TwoLeadECG": 23,
        "Wafer": 128,
        "CBF": 30,
        "Plane": 16,
    }

    CFRL_BATCH_SIZE = DATASET_BATCH_SIZES.get(args.dataset)
    BATCH_SIZE_AE = CFRL_BATCH_SIZE
    BATCH_SIZE = CFRL_BATCH_SIZE

    X_train, Y_train, X_test, Y_test = readUCR(args.dataset)
    X_train, Y_train, X_test, Y_test, onehot_encoder = prepare_data(X_train, Y_train, X_test, Y_test)

    # Get dataset info
    nb_timesteps = X_train.shape[1]
    nb_features = 1
    unique_labels = np.unique(np.concatenate((np.argmax(Y_train, axis=1), np.argmax(Y_test, axis=1))))
    num_classes = len(unique_labels)
    print(f"Number of Classes: {num_classes}")

    # Store dataset info
    metrics.update({
        'num_classes': num_classes,
        'nb_timesteps': nb_timesteps,
        'train_samples': X_train.shape[0],
        'test_samples': X_test.shape[0]
    })

    # Initialize wandb
    wandb.login(key="xxx")
    run_name = f"CFRL_{args.dataset}" + ("_no_replay" if args.no_replay else "")
    wandb.init(project=WANDB_PROJECT, entity="xxx", name=run_name)

    # Create datasets
    trainset_classifier, testset_classifier, trainset_ae, testset_ae = create_datasets(
        X_train, Y_train, X_test, Y_test, BATCH_SIZE
    )

    # Train or load classifier
    print("Setting up classifier...")
    classifier = train_or_load_classifier(X_train, Y_train, X_test, Y_test, num_classes, args.dataset)

    # Store classifier metrics
    loss, accuracy = classifier.evaluate(X_test, Y_test, verbose=0)
    metrics.update({
        'classifier_accuracy': accuracy,
        'classifier_loss': loss
    })

    # Define predictor function
    def predictor(X: np.ndarray):
        return classifier(X, training=False).numpy()

    # Train or load autoencoder and capture metrics
    print("Setting up autoencoder...")
    ae, ae_metrics = train_or_load_autoencoder_with_metrics(trainset_ae, testset_ae, nb_timesteps, nb_features, args.dataset)
    metrics.update(ae_metrics)

    # Visualize autoencoder reconstruction
    visualize_autoencoder_reconstruction(ae, X_test, args.dataset)

    # Calculate embedding space separation
    embedding_separation = calculate_embedding_separation(ae, X_test, Y_test)
    metrics['embedding_separation'] = embedding_separation

    # Generate targets for training data
    print("Generating targets for training data...")
    Y_t_train, Y_t_int_train, Y_m_train = generate_targets(X_train, predictor, num_classes, onehot_encoder)

    print(f"Original predictions (train): {Y_m_train[:10]}")
    print(f"Target predictions (train): {Y_t_int_train[:10]}")
    print(f"Are they different? {np.all(Y_m_train != Y_t_int_train)}")

    # Define explainer with callbacks
    print("Setting up CFRL explainer...")
    callbacks = [
        RewardCallback(),
        ImprovedTimeSeriesCallback(num_classes),
        LossCallback(),
        SuccessRateCallback()
    ]

    explainer = CounterfactualRL(
        predictor=predictor,
        encoder=ae.encoder,
        decoder=ae.decoder,
        latent_dim=LATENT_DIM,
        coeff_sparsity=COEFF_SPARSITY,
        coeff_consistency=COEFF_CONSISTENCY,
        train_steps=TRAIN_STEPS,
        batch_size=CFRL_BATCH_SIZE,
        backend="tensorflow",
        callbacks=callbacks,
        use_replay=(not args.no_replay),
    )

    # Validate batch size
    assert X_train.shape[0] >= CFRL_BATCH_SIZE, "Batch size must not exceed number of samples in X_train."

    # Fit explainer with timing
    print("Fitting explainer with training targets...")
    training_start_time = time.time()
    explainer.fit(X=X_train)
    training_end_time = time.time()
    training_time = training_end_time - training_start_time
    print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

    # Store training time metrics
    metrics.update({
        'training_time_seconds': training_time,
        'training_time_minutes': training_time / 60,
        'training_time_hours': training_time / 3600
    })

    # Log training time to wandb
    wandb.log({
        'training_time_seconds': training_time,
        'training_time_minutes': training_time / 60,
        'training_time_hours': training_time / 3600
    })

    # Generate targets for test data and create explanations
    print("Generating explanations for test data...")
    Y_t_test, Y_t_int_test, Y_m_test = generate_targets(X_test, predictor, num_classes, onehot_encoder)

    # Inference with timing
    inference_start_time = time.time()
    explanation = explainer.explain(X_test, Y_t=Y_t_test, batch_size=10)
    inference_end_time = time.time()
    inference_time_total = inference_end_time - inference_start_time
    inference_time_per_instance = inference_time_total / len(X_test)

    print(f"Total inference time: {inference_time_total:.2f} seconds ({inference_time_total/60:.2f} minutes)")
    print(f"Average inference time per instance: {inference_time_per_instance:.4f} seconds")

    # Store inference time metrics
    metrics.update({
        'inference_time_total_seconds': inference_time_total,
        'inference_time_total_minutes': inference_time_total / 60,
        'inference_time_per_instance_seconds': inference_time_per_instance,
        'inference_time_per_instance_ms': inference_time_per_instance * 1000
    })

    # Log inference time to wandb
    wandb.log({
        'inference_time_total_seconds': inference_time_total,
        'inference_time_total_minutes': inference_time_total / 60,
        'inference_time_per_instance_seconds': inference_time_per_instance,
        'inference_time_per_instance_ms': inference_time_per_instance * 1000,
        'test_samples': len(X_test)
    })

    # Save results
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    cf_X = explanation.data['cf']['X']
    cf_valid = (np.argmax(predictor(cf_X), axis=1) == np.argmax(Y_t_test, axis=1))
    cf_X_final = np.where(cf_valid[:, None, None], cf_X, -np.ones_like(cf_X))
    np.save(os.path.join(DEFAULT_OUTPUT_DIR, f'counterfactuals_{args.dataset}.npy'), cf_X_final)

    # Evaluate and get final metrics
    final_metrics = evaluate_and_get_metrics(explanation, Y_t_test, args.dataset, predictor)
    metrics.update(final_metrics)

    # Compute additional metrics for valid counterfactuals only
    cf_X = explanation.data['cf']['X']
    orig_X = explanation.data['orig']['X']
    valid_mask = (np.argmax(predictor(cf_X), axis=1) == np.argmax(Y_t_test, axis=1))
    cf_X_valid = cf_X[valid_mask]
    orig_X_valid = orig_X[valid_mask]

    l1s, l2s, l_infs, sparsities, segnums = [], [], [], [], []
    for i in range(len(cf_X_valid)):
        l1, l2, l_inf, sparsity, segnum = getmetrics(cf_X_valid[i].squeeze(), orig_X_valid[i].squeeze())
        l1s.append(l1)
        l2s.append(l2)
        l_infs.append(l_inf)
        sparsities.append(sparsity)
        segnums.append(segnum)
    metrics.update({
        'L1_mean': np.mean(l1s) if l1s else np.nan,
        'L1_std': np.std(l1s) if l1s else np.nan,
        'L2_mean': np.mean(l2s) if l2s else np.nan,
        'L2_std': np.std(l2s) if l2s else np.nan,
        'L_inf_mean': np.mean(l_infs) if l_infs else np.nan,
        'L_inf_std': np.std(l_infs) if l_infs else np.nan,
        'Sparsity_mean': np.mean(sparsities) if sparsities else np.nan,
        'Sparsity_std': np.std(sparsities) if sparsities else np.nan,
        'Segment_mean': np.mean(segnums) if segnums else np.nan,
        'Segment_std': np.std(segnums) if segnums else np.nan
    })

    # OOD metrics (using LOF, SVM, IsolationForest) for valid counterfactuals only
    try:
        OOD_svm, OOD_lof, mean_OOD_ifo = cf_ood(orig_X_valid.squeeze(), cf_X_valid.squeeze())
        metrics.update({
            'OOD_svm': OOD_svm,
            'OOD_lof': OOD_lof,
            'mean_OOD_ifo': mean_OOD_ifo
        })
    except Exception as e:
        print(f"OOD metric calculation failed: {e}")

    # Calculate overall runtime
    overall_end_time = time.time()
    overall_runtime = overall_end_time - overall_start_time
    print(f"\nTotal runtime: {overall_runtime:.2f} seconds ({overall_runtime/60:.2f} minutes, {overall_runtime/3600:.2f} hours)")

    # Store overall runtime metrics
    metrics.update({
        'total_runtime_seconds': overall_runtime,
        'total_runtime_minutes': overall_runtime / 60,
        'total_runtime_hours': overall_runtime / 3600
    })

    # Log overall runtime to wandb
    wandb.log({
        'total_runtime_seconds': overall_runtime,
        'total_runtime_minutes': overall_runtime / 60,
        'total_runtime_hours': overall_runtime / 3600
    })

    # Log efficiency metrics to wandb
    wandb.log({
        'training_to_inference_ratio': training_time / inference_time_total if inference_time_total > 0 else 0,
        'training_per_step_ms': (training_time / TRAIN_STEPS) * 1000,
        'throughput_instances_per_second': len(X_test) / inference_time_total if inference_time_total > 0 else 0
    })

    # Save all metrics to CSV
    save_metrics_to_csv(args.dataset, metrics)
    # Cleanup
    wandb.finish()
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
