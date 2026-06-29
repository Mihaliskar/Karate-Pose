from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# SETTINGS
# -----------------------------

TRANSFORMED_FOLDER = Path("transformed")
EDM_FOLDER = Path("edm")
OUTPUT_FOLDER = Path("comparison_output")

# Choose one:
# "error"      -> total error between files
# "rmse"       -> average error size
# "similarity" -> 1.0 means identical, lower means more different
OUTPUT_TYPE = "error"
#OUTPUT_TYPE = "similarity"

# -----------------------------
# LOADING FUNCTIONS
# -----------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_transformed(path):
    """
    Loads a transformed skeleton JSON.

    Expected format:
    {
        "pelvis": [x, y, z],
        "left_hip": [x, y, z],
        ...
    }
    """
    data = load_json(path)

    if not isinstance(data, dict):
        raise ValueError(f"{path.name} is not a joint dictionary.")

    cleaned = {}

    for joint_name, coords in data.items():
        arr = np.array(coords, dtype=float)

        if arr.shape != (3,):
            raise ValueError(
                f"{path.name}: joint '{joint_name}' does not contain exactly 3 values."
            )

        cleaned[joint_name] = arr

    return cleaned


def load_edm(path):
    """
    Loads an EDM JSON.

    Expected format:
    [
        [0.0, 0.1, ...],
        [0.1, 0.0, ...],
        ...
    ]
    """
    matrix = np.array(load_json(path), dtype=float)

    if matrix.ndim != 2:
        raise ValueError(f"{path.name} is not a 2D matrix.")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{path.name} is not a square matrix.")

    return matrix


# -----------------------------
# COMPARISON FUNCTIONS
# -----------------------------

def transformed_error(file_a, file_b):
    """
    Compares two already-aligned transformed skeleton files.

    Returns:
    - total_error: sum of Euclidean distances between matching joints
    - rmse: root mean squared coordinate error
    """
    common_joints = sorted(set(file_a.keys()) & set(file_b.keys()))

    if not common_joints:
        raise ValueError("The two transformed files have no joints in common.")

    A = np.array([file_a[joint] for joint in common_joints], dtype=float)
    B = np.array([file_b[joint] for joint in common_joints], dtype=float)

    joint_distances = np.linalg.norm(A - B, axis=1)

    total_error = float(np.sum(joint_distances))
    rmse = float(np.sqrt(np.mean((A - B) ** 2)))

    return total_error, rmse


def edm_error(matrix_a, matrix_b):
    """
    Compares two EDM matrices.

    Returns:
    - total_error: sum of absolute differences between matrix values
    - rmse: root mean squared matrix error
    """
    if matrix_a.shape != matrix_b.shape:
        raise ValueError(
            f"EDM matrix shapes do not match: {matrix_a.shape} vs {matrix_b.shape}"
        )

    diff = matrix_a - matrix_b

    total_error = float(np.sum(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    return total_error, rmse


def rmse_to_similarity(rmse):
    """
    Converts RMSE error to a similarity score.

    1.0 = identical
    closer to 0.0 = more different
    """
    return 1.0 / (1.0 + rmse)


# -----------------------------
# PAIRWISE MATRIX CREATION
# -----------------------------

def build_pairwise_matrix(files, loader, error_function, output_type):
    names = [path.stem for path in files]
    loaded_files = [loader(path) for path in files]

    matrix = np.zeros((len(files), len(files)), dtype=float)

    for i in range(len(files)):
        for j in range(len(files)):
            total_error, rmse = error_function(loaded_files[i], loaded_files[j])

            if output_type == "error":
                value = total_error
            elif output_type == "rmse":
                value = rmse
            elif output_type == "similarity":
                value = rmse_to_similarity(rmse)
            else:
                raise ValueError(
                    "OUTPUT_TYPE must be 'error', 'rmse', or 'similarity'."
                )

            matrix[i, j] = value

    return pd.DataFrame(matrix, index=names, columns=names)


# -----------------------------
# HEATMAP / CHART OUTPUT
# -----------------------------

def save_heatmap(df, title, output_path):
    """
    Saves a pairwise comparison heatmap.

    Rows = file A
    Columns = file B
    Middle values = comparison result
    """
    size = max(6, len(df) * 0.8)

    fig, ax = plt.subplots(figsize=(size, size))

    image = ax.imshow(df.values)
    plt.colorbar(image, ax=ax)

    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index)))

    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticklabels(df.index)

    ax.set_title(title)

    # Show numbers inside cells if there are not too many files
    if len(df) <= 20:
        for row in range(len(df.index)):
            for col in range(len(df.columns)):
                ax.text(
                    col,
                    row,
                    f"{df.iloc[row, col]:.3f}",
                    ha="center",
                    va="center",
                )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# -----------------------------
# FOLDER COMPARISON
# -----------------------------

def compare_folder(folder, loader, error_function, output_prefix, output_type):
    files = sorted(folder.glob("*.json"))

    if len(files) == 0:
        print(f"No JSON files found in folder: {folder}")
        return

    if len(files) == 1:
        print(f"Only one JSON file found in folder: {folder}")
        print("A comparison matrix needs at least two files.")
        return

    df = build_pairwise_matrix(
        files=files,
        loader=loader,
        error_function=error_function,
        output_type=output_type,
    )

    csv_path = OUTPUT_FOLDER / f"{output_prefix}_{output_type}_matrix.csv"
    png_path = OUTPUT_FOLDER / f"{output_prefix}_{output_type}_heatmap.png"

    df.to_csv(csv_path)

    save_heatmap(
        df=df,
        title=f"{output_prefix} {output_type} comparison",
        output_path=png_path,
    )

    print()
    print(f"{output_prefix.upper()} {output_type.upper()} MATRIX")
    print(df.round(4))
    print(f"Saved CSV: {csv_path}")
    print(f"Saved heatmap: {png_path}")


# -----------------------------
# MAIN
# -----------------------------

def main():
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    compare_folder(
        folder=TRANSFORMED_FOLDER,
        loader=load_transformed,
        error_function=transformed_error,
        output_prefix="transformed",
        output_type=OUTPUT_TYPE,
    )

    compare_folder(
        folder=EDM_FOLDER,
        loader=load_edm,
        error_function=edm_error,
        output_prefix="edm",
        output_type=OUTPUT_TYPE,
    )


if __name__ == "__main__":
    main()