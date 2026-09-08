import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix



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

def train_ovr_linear(X_train, y_train):
    binary_svm = LinearSVC(
        C=1.0,
        random_state=42,
        max_iter=10_000,
    )

    model = OneVsRestClassifier(binary_svm)
    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_validation, y_validation):
    predictions = model.predict(X_validation)

    accuracy = accuracy_score(y_validation, predictions)
    macro_f1 = f1_score(
        y_validation,
        predictions,
        average="macro",
    )

    return predictions, accuracy, macro_f1

def print_detailed_results(y_validation, predictions, class_names):
    print("\nClassification report:")
    print(
        classification_report(
            y_validation,
            predictions,
            target_names=class_names,
            digits=4,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_validation, predictions))

def main():
    with np.load("features/features.npz") as features:
        X_train, y_train = load_training_data(features)

        model = train_ovr_linear(X_train, y_train)

        print("OVR linear SVM training completed.")

        X_validation, y_validation = load_validation_data(features)

        predictions, accuracy, macro_f1 = evaluate_model(model, X_validation, y_validation)

        print(f"Validation accuracy: {accuracy:.4f}")
        print(f"Validation macro F1: {macro_f1:.4f}")

        class_names = features["class_names"]

        print_detailed_results(y_validation, predictions, class_names)



        X_test, y_test = load_testing_data(features)

        features.close()


if __name__ == "__main__":
    main()