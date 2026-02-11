import os
import sys

# Ensure repository root is on sys.path so 'alibi' can be imported without installation
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
from alibi.utils.frameworks import has_tensorflow

if has_tensorflow:
    import tensorflow as tf
    from alibi.explainers.cfrl_base import CounterfactualRL


def _build_simple_autoencoder(input_dim: int, latent_dim: int):
    encoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(latent_dim, activation='linear'),
    ])
    decoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(latent_dim,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(input_dim, activation='linear'),
    ])
    return encoder, decoder


def _build_predictor(input_dim: int, num_classes: int = 2):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])
    # Random init; no training required for this sanity test.
    return lambda X: model(X).numpy()


def test_use_replay_true_false():
    if not has_tensorflow:
        print('TensorFlow not available; skipping use_replay toggle test.')
        return

    input_dim = 8
    latent_dim = 4
    X = np.random.randn(120, input_dim).astype(np.float32)

    encoder, decoder = _build_simple_autoencoder(input_dim=input_dim, latent_dim=latent_dim)
    predictor = _build_predictor(input_dim=input_dim)

    # Test with experience replay enabled.
    expl_replay = CounterfactualRL(predictor=predictor,
                                   encoder=encoder,
                                   decoder=decoder,
                                   coeff_sparsity=0.1,
                                   coeff_consistency=0.1,
                                   latent_dim=latent_dim,
                                   backend='tensorflow',
                                   train_steps=6,
                                   batch_size=12,
                                   update_every=1,
                                   update_after=2,
                                   use_replay=True)
    expl_replay.fit(X)
    assert expl_replay.params['use_replay'] is True

    # Test with experience replay disabled.
    expl_no_replay = CounterfactualRL(predictor=predictor,
                                      encoder=encoder,
                                      decoder=decoder,
                                      coeff_sparsity=0.1,
                                      coeff_consistency=0.1,
                                      latent_dim=latent_dim,
                                      backend='tensorflow',
                                      train_steps=6,
                                      batch_size=12,
                                      update_every=1,
                                      update_after=3,  # should be ignored
                                      use_replay=False)
    expl_no_replay.fit(X)
    assert expl_no_replay.params['use_replay'] is False

    print('use_replay toggle test passed for both modes.')


if __name__ == '__main__':
    test_use_replay_true_false()
