#!/usr/bin/env python3
"""Validate karate-pose EDM files and build features/features.npz.

The metadata CSV must contain:
    image_path, edm_path, class_name, person_id, group_id

Each EDM JSON file must contain a square matrix directly as a nested list.
For a 22 x 22 EDM, the script extracts 231 unique upper-triangle distances.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


REQUIRED_COLUMNS = {
    "image_path",
    "edm_path",
    "class_name",
    "person_id",
    "group_id",
}
SYMMETRY_TOLERANCE = 1e-5
DIAGONAL_TOLERANCE = 1e-5
NEGATIVE_TOLERANCE = 1e-6
ZERO_TOLERANCE = 1e-8


def parse_arguments() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Build raw and normalized feature vectors from EDM JSON files."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_root,
        help=(
            "Directory containing metadata/, edms/ and features/. "
            "By default this is the parent of src/."
        ),
    )
    return parser.parse_args()


def read_metadata(metadata_path: Path) -> list[dict[str, str]]:
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")

    with metadata_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = set(reader.fieldnames or [])
        missing_columns = REQUIRED_COLUMNS - columns

        if missing_columns:
            raise ValueError(
                f"Metadata CSV is missing columns: {sorted(missing_columns)}"
            )

        # Your current CSV contains blank lines. Ignore rows with no values.
        rows = [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
            if any((value or "").strip() for value in row.values())
        ]

    if not rows:
        raise ValueError("Metadata CSV contains no non-empty sample rows")

    return rows


def load_and_validate_edm(edm_path: Path) -> np.ndarray:
    if not edm_path.is_file():
        raise FileNotFoundError(f"EDM file not found: {edm_path}")

    if edm_path.suffix.lower() != ".json":
        raise ValueError(f"Expected a JSON EDM, got: {edm_path.suffix}")

    try:
        with edm_path.open("r", encoding="utf-8") as json_file:
            data = json.load(json_file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON: {error}") from error

    # The user's JSON files contain the matrix directly, not under an "edm" key.
    try:
        edm = np.asarray(data, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("EDM does not contain a numeric matrix") from error

    if edm.ndim != 2:
        raise ValueError(f"EDM must be two-dimensional, got shape {edm.shape}")

    if edm.shape[0] != edm.shape[1]:
        raise ValueError(f"EDM must be square, got shape {edm.shape}")

    if edm.shape[0] < 2:
        raise ValueError(f"EDM is too small: {edm.shape}")

    if not np.all(np.isfinite(edm)):
        invalid_count = int(np.size(edm) - np.count_nonzero(np.isfinite(edm)))
        raise ValueError(f"EDM contains {invalid_count} NaN or infinite values")

    maximum_symmetry_error = float(np.max(np.abs(edm - edm.T)))
    if maximum_symmetry_error > SYMMETRY_TOLERANCE:
        raise ValueError(
            "EDM is not symmetric; maximum difference is "
            f"{maximum_symmetry_error:.8g}"
        )

    maximum_diagonal_value = float(np.max(np.abs(np.diag(edm))))
    if maximum_diagonal_value > DIAGONAL_TOLERANCE:
        raise ValueError(
            "EDM diagonal is not zero; maximum absolute value is "
            f"{maximum_diagonal_value:.8g}"
        )

    minimum_value = float(np.min(edm))
    if minimum_value < -NEGATIVE_TOLERANCE:
        raise ValueError(f"EDM contains a negative distance: {minimum_value}")

    # Remove harmless floating-point errors close to zero.
    edm[np.abs(edm) < ZERO_TOLERANCE] = 0.0
    edm[edm < 0.0] = 0.0

    return edm


def vectorize_upper_triangle(edm: np.ndarray) -> np.ndarray:
    """Return every unique distance once, excluding the zero diagonal."""
    indices = np.triu_indices_from(edm, k=1)
    return edm[indices].astype(np.float32, copy=False)


def normalize_edm(edm: np.ndarray) -> np.ndarray:
    """Remove overall body/image scale by dividing by the largest distance."""
    maximum_distance = float(np.max(edm))

    if maximum_distance <= ZERO_TOLERANCE:
        raise ValueError("EDM maximum distance is zero")

    return edm / maximum_distance


def resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def main() -> None:
    arguments = parse_arguments()
    project_root = arguments.project_root.expanduser().resolve()
    metadata_path = project_root / "metadata" / "dataset.csv"
    output_path = project_root / "features" / "features.npz"

    rows = read_metadata(metadata_path)

    class_names = sorted({row["class_name"] for row in rows if row["class_name"]})
    if not class_names:
        raise ValueError("No class names were found in the metadata CSV")

    class_to_index = {
        class_name: class_index
        for class_index, class_name in enumerate(class_names)
    }

    raw_features: list[np.ndarray] = []
    normalized_features: list[np.ndarray] = []
    labels: list[int] = []
    image_paths: list[str] = []
    edm_paths: list[str] = []
    person_ids: list[str] = []
    group_ids: list[str] = []

    expected_matrix_shape: tuple[int, int] | None = None
    seen_image_paths: set[str] = set()

    for csv_line, row in enumerate(rows, start=2):
        image_path = row["image_path"]
        edm_relative_path = row["edm_path"]
        class_name = row["class_name"]

        if not image_path or not edm_relative_path or not class_name:
            raise ValueError(
                f"CSV line {csv_line} is missing image_path, edm_path or class_name"
            )

        if image_path in seen_image_paths:
            raise ValueError(f"Duplicate image_path on CSV line {csv_line}: {image_path}")
        seen_image_paths.add(image_path)

        if class_name not in class_to_index:
            raise ValueError(f"Unknown class on CSV line {csv_line}: {class_name}")

        edm_path = resolve_project_path(project_root, edm_relative_path)

        try:
            edm = load_and_validate_edm(edm_path)
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"Failed to process CSV line {csv_line} ({edm_relative_path}): {error}"
            ) from error

        if expected_matrix_shape is None:
            expected_matrix_shape = edm.shape
        elif edm.shape != expected_matrix_shape:
            raise ValueError(
                f"Inconsistent EDM shape on CSV line {csv_line}: "
                f"expected {expected_matrix_shape}, got {edm.shape}"
            )

        raw_features.append(vectorize_upper_triangle(edm))
        normalized_features.append(vectorize_upper_triangle(normalize_edm(edm)))
        labels.append(class_to_index[class_name])
        image_paths.append(image_path)
        edm_paths.append(edm_relative_path)
        person_ids.append(row["person_id"])
        group_ids.append(row["group_id"])

    X_raw = np.stack(raw_features).astype(np.float32, copy=False)
    X_normalized = np.stack(normalized_features).astype(np.float32, copy=False)
    y = np.asarray(labels, dtype=np.int64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        # X defaults to normalized features for convenient use by later scripts.
        X=X_normalized,
        X_raw=X_raw,
        X_normalized=X_normalized,
        y=y,
        paths=np.asarray(image_paths, dtype=str),
        edm_paths=np.asarray(edm_paths, dtype=str),
        class_names=np.asarray(class_names, dtype=str),
        person_ids=np.asarray(person_ids, dtype=str),
        groups=np.asarray(group_ids, dtype=str),
    )

    print(f"Created: {output_path}")
    print(f"Samples: {len(y)}")
    print(f"EDM shape: {expected_matrix_shape}")
    print(f"Feature shape: {X_raw.shape}")
    print(f"NaN values: {int(np.isnan(X_raw).sum())}")
    print("\nClass mapping and counts:")

    counts = Counter(y.tolist())
    for class_index, class_name in enumerate(class_names):
        print(f"  {class_index}: {class_name}: {counts[class_index]}")

    empty_people = sum(not value for value in person_ids)
    empty_groups = sum(not value for value in group_ids)
    if empty_people or empty_groups:
        print("\nMetadata warnings:")
        print(f"  Empty person_id values: {empty_people}")
        print(f"  Empty group_id values: {empty_groups}")


if __name__ == "__main__":
    main()
