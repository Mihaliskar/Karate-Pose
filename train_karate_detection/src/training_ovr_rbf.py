import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
import joblib



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

def train_ovr_rbf(X_train, y_train):
    binary_svm = SVC(
        kernel="rbf",
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

def save_model(model, class_names, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_data = {
        "model": model,
        "class_names": class_names,
        "feature_type": "X",
    }

    joblib.dump(model_data, output_path)

def save_validation_results(y_validation, predictions, accuracy, macro_f1, class_names, output_directory):
    output_directory.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        y_validation,
        predictions,
        target_names=class_names,
        digits=4,
    )

    results_path = output_directory / "validation_results.txt"

    with results_path.open("w", encoding="utf-8") as results_file:
        results_file.write("OVR RBF SVM\n")
        results_file.write("========================\n")
        results_file.write(f"Validation accuracy: {accuracy:.4f}\n")
        results_file.write(f"Validation macro F1: {macro_f1:.4f}\n\n")
        results_file.write("Classification report:\n")
        results_file.write(report)

def save_confusion_matrix_csv(y_validation, predictions, output_path):
    matrix = confusion_matrix(y_validation, predictions)

    np.savetxt(
        output_path,
        matrix,
        delimiter=",",
        fmt="%d",
    )

def save_confusion_matrix_heatmap(y_validation, predictions, class_names, output_path):
    matrix = confusion_matrix(y_validation, predictions)

    figure, axis = plt.subplots(figsize=(11, 9))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )

    display.plot(
        ax=axis,
        cmap="Blues",
        values_format="d",
        colorbar=True,
    )

    axis.set_title("OVR RBF SVM — Validation Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

def save_test_results(y_test, predictions, accuracy, macro_f1, class_names, output_directory):
    labels = np.arange(len(class_names))

    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    output_path = output_directory / "test_results.txt"

    with output_path.open("w", encoding="utf-8") as results_file:
        results_file.write("OVR RBF SVM — Test Results\n")
        results_file.write("=============================\n")
        results_file.write(f"Test accuracy: {accuracy:.4f}\n")
        results_file.write(f"Test macro F1: {macro_f1:.4f}\n\n")
        results_file.write("Classification report:\n")
        results_file.write(report)

def save_test_confusion_matrix_heatmap(y_test, predictions, class_names, output_path):
    labels = np.arange(len(class_names))

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    figure, axis = plt.subplots(figsize=(11, 9))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )

    display.plot(
        ax=axis,
        cmap="Blues",
        values_format="d",
        colorbar=True,
    )

    axis.set_title("OVR RBF SVM — Test Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

def load_model(model_path):
    model_data = joblib.load(model_path)

    model = model_data["model"]
    class_names = model_data["class_names"]
    feature_type = model_data["feature_type"]

    return model, class_names, feature_type

def main():
    output_directory = Path("results/ovr_rbf")
    model_path = output_directory / "model.joblib"

    # ---------------------------------------------------------
    # Train and validate the model
    # ---------------------------------------------------------

    with np.load("features/features.npz") as features:
        X_train, y_train = load_training_data(features)
        X_validation, y_validation = load_validation_data(features)
        class_names = features["class_names"].copy()

        model = train_ovr_rbf(X_train, y_train)

        validation_predictions, validation_accuracy, validation_macro_f1 = (
            evaluate_model(
                model,
                X_validation,
                y_validation,
            )
        )

    print("OVR RBF SVM training completed.")
    print(f"Validation accuracy: {validation_accuracy:.4f}")
    print(f"Validation macro F1: {validation_macro_f1:.4f}")

    print_detailed_results(
        y_validation,
        validation_predictions,
        class_names,
    )

    save_model(
        model,
        class_names,
        model_path,
    )

    save_validation_results(
        y_validation,
        validation_predictions,
        validation_accuracy,
        validation_macro_f1,
        class_names,
        output_directory,
    )

    save_confusion_matrix_csv(
        y_validation,
        validation_predictions,
        output_directory / "validation_confusion_matrix.csv",
    )

    save_confusion_matrix_heatmap(
        y_validation,
        validation_predictions,
        class_names,
        output_directory / "validation_confusion_matrix.png",
    )

    # ---------------------------------------------------------
    # Load the saved model
    # ---------------------------------------------------------

    model_data = joblib.load(model_path)

    loaded_model = model_data["model"]
    loaded_class_names = model_data["class_names"]
    feature_type = model_data["feature_type"]

    print("\nSaved model loaded successfully.")
    print(f"Feature type: {feature_type}")

    # ---------------------------------------------------------
    # Load and evaluate the test set
    # ---------------------------------------------------------

    with np.load("features/features.npz") as features:
        test_indices = np.load("splits/test_indices.npy")

        X_test = features[feature_type][test_indices]
        y_test = features["y"][test_indices]

    test_predictions, test_accuracy, test_macro_f1 = evaluate_model(
        loaded_model,
        X_test,
        y_test,
    )

    print(f"\nTest samples: {len(y_test)}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test macro F1: {test_macro_f1:.4f}")

    print_detailed_results(
        y_test,
        test_predictions,
        loaded_class_names,
    )

    # ---------------------------------------------------------
    # Save the test results
    # ---------------------------------------------------------

    save_test_results(
        y_test,
        test_predictions,
        test_accuracy,
        test_macro_f1,
        loaded_class_names,
        output_directory,
    )

    save_confusion_matrix_csv(
        y_test,
        test_predictions,
        output_directory / "test_confusion_matrix.csv",
    )

    save_test_confusion_matrix_heatmap(
        y_test,
        test_predictions,
        loaded_class_names,
        output_directory / "test_confusion_matrix.png",
    )

    print(f"\nModel and results saved to: {output_directory}")


if __name__ == "__main__":
    main()
