import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
import joblib
import argparse

def load_data(type, features):
    print(f"Loading {type} data")

    indices = np.load(f"splits/{type}_indices.npy")
    
    X = features["X"][indices]
    y = features["y"][indices]
    
    return X, y

def train_model(type, kernel_type, c, X_train, y_train):
    binary_svm = SVC(
        kernel=kernel_type, #linear or rbf
        C=c,
        random_state=42,
        max_iter=10_000,
    )

    model = None
    if (type == 0): #ovr=0, ovo=1
        model = OneVsRestClassifier(binary_svm)
    elif (type == 1):
        model = OneVsOneClassifier(binary_svm)
    else:
        return None
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

def save_results(type, name, y, predictions, accuracy, macro_f1, class_names, output_directory):
    output_directory.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        y,
        predictions,
        target_names=class_names,
        digits=4,
    )

    results_path = output_directory / (type + "_results.txt")

    with results_path.open("w", encoding="utf-8") as results_file:
        results_file.write(name + "\n")
        results_file.write("========================\n")
        results_file.write(f"{type} accuracy: {accuracy:.4f}\n")
        results_file.write(f"{type} macro F1: {macro_f1:.4f}\n\n")
        results_file.write("Classification report:\n")
        results_file.write(report)

def save_confusion_matrix_csv(y, predictions, output_path):
    matrix = confusion_matrix(y, predictions)

    np.savetxt(
        output_path,
        matrix,
        delimiter=",",
        fmt="%d",
    )

def save_confusion_matrix_heatmap(name, type, y, predictions, class_names, output_path):
    matrix = confusion_matrix(y, predictions)

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

    axis.set_title(name + " — " + type + " Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

def save_model(model, class_names, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_data = {
        "model": model,
        "class_names": class_names,
        "feature_type": "X",
    }

    joblib.dump(model_data, output_path)

def load_model(model_path):
    model_data = joblib.load(model_path)

    model = model_data["model"]
    class_names = model_data["class_names"]
    feature_type = model_data["feature_type"]

    return model, class_names, feature_type

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train or test a model."
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model."
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Test the model."
    )

    parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Output directory.",
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Model input/output path.",
    )

    parser.add_argument(
        "--type",
        choices=[
            "ovr_linear",
            "ovr_rbf",
            "ovo_linear",
            "ovo_rbf",
        ],
        required=True,
        help="Model type to train or test.",
    )

    return parser.parse_args()

def main():
    args = parse_arguments()
    train = args.train
    test = args.test
    model_type = args.type
    output_directory = args.output if args.output is not None else Path("results") / model_type
    model_path = args.input if args.input is not None else output_directory / "model.joblib"

    if (train == False and test == False):
        print("No action selected")
        print("Exiting...")
        return

    with np.load("features/features.npz") as features:
        class_names = features["class_names"].copy()

        match model_type:
            case "ovr_linear":
                model_name = "OVR Linear"
                strategy = 0
                kernel = "linear"
            case "ovr_rbf":
                model_name = "OVR RBF"
                strategy = 0
                kernel = "rbf"
                    
            case "ovo_linear":
                model_name = "OVO Linear"
                strategy = 1
                kernel = "linear"
                    
            case "ovo_rbf":
                model_name = "OVO RBF"
                strategy = 1
                kernel = "rbf"
                    
            case _:
                print(f"Unknown model type: {args.type}")
                return
            
        if (train):
            X_train, y_train = load_data("train", features)
            X_validation, y_validation = load_data("validation", features)

            print(f"Starting training of {model_name} SVM")
            model = train_model(strategy, kernel, 1.0, X_train, y_train)

            validation_predictions, validation_accuracy, validation_macro_f1 = evaluate_model(model, X_validation, y_validation)

            print("Training completed.")
            print(f"Validation accuracy: {validation_accuracy:.4f}")
            print(f"Validation macro F1: {validation_macro_f1:.4f}")

            print("Saving Model and Results")

            print_detailed_results(y_validation, validation_predictions,class_names)
            
            save_model(model, class_names, model_path)

            save_results("validation", model_name, y_validation, validation_predictions, validation_accuracy, validation_macro_f1, class_names, output_directory)
            
            save_confusion_matrix_csv(y_validation, validation_predictions, output_directory / "validation_confusion_matrix.csv")
            
            save_confusion_matrix_heatmap(f"{model_name} SVM", "Validation", y_validation, validation_predictions, class_names, output_directory / "validation_confusion_matrix.png")

        if (test):
            print("Starting Model Testing")
            X_test, y_test = load_data("test", features)

            loaded_model, loaded_class_names, feature_type = load_model(model_path)

            print("\nSaved model loaded successfully.")
            print(f"Feature type: {feature_type}")

            test_predictions, test_accuracy, test_macro_f1 = evaluate_model(loaded_model, X_test, y_test)

            print("Testing Complete")
            print(f"\nTest samples: {len(y_test)}")
            print(f"Test accuracy: {test_accuracy:.4f}")
            print(f"Test macro F1: {test_macro_f1:.4f}")
            print("Saving Testing Results")

            save_results("test", model_name, y_test, test_predictions, test_accuracy, test_macro_f1, class_names, output_directory)

            save_confusion_matrix_csv(y_test, test_predictions, output_directory / "test_confusion_matrix.csv")
            
            save_confusion_matrix_heatmap(f"{model_name} SVM", "Test", y_test, test_predictions, loaded_class_names, output_directory / "test_confusion_matrix.png")
            
        print("Process Complete")

if __name__ == "__main__":
    main()