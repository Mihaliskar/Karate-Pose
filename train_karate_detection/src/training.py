import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt



def load_training_data(features):
    train_indices = np.load("splits/train_indices.npy")

    X_train = features["X"][train_indices]
    y_train = features["y"][train_indices]

    return X_train, y_train


def load_validation_data(features):
    validation_indices = np.load("splits/validation_indices.npy")

    X_validation = features["X"][validation_indices]
    y_validation = features["y"][validation_indices]

    return X_validation, y_validation


def load_testing_data(features):
    test_indices = np.load("splits/test_indices.npy")

    X_test = features["X"][test_indices]
    y_test = features["y"][test_indices]

    return X_test, y_test


def main():
    with np.load("features/features.npz") as features:
        X_train, y_train = load_training_data(features)
        X_validation, y_validation = load_validation_data(features)
        X_test, y_test = load_testing_data(features)

        print(f"Training:   X={X_train.shape}, y={y_train.shape}")
        print(f"Validation: X={X_validation.shape}, y={y_validation.shape}")
        print(f"Testing:    X={X_test.shape}, y={y_test.shape}")


if __name__ == "__main__":
    main()