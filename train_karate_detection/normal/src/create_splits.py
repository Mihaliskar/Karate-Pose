#!/usr/bin/env python3
"""Create reproducible group-aware train, validation, and test splits.

This script reads features/features.npz and writes split indices to splits/.
Samples with the same (person_id, group_id) are always kept in one split.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


REQUIRED_ARRAYS = {
    "y",
    "paths",
    "class_names",
    "person_ids",
    "groups",
}


def parse_arguments() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description=(
            "Create approximately 70/15/15 train, validation and test splits "
            "while keeping recording groups together."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_root,
        help=(
            "Directory containing features/ and splits/. "
            "By default this is the parent of src/."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to create the folds (default: 42).",
    )
    return parser.parse_args()


def load_feature_metadata(
    feature_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not feature_path.is_file():
        raise FileNotFoundError(f"Feature archive not found: {feature_path}")

    with np.load(feature_path, allow_pickle=False) as archive:
        missing_arrays = REQUIRED_ARRAYS - set(archive.files)
        if missing_arrays:
            raise ValueError(
                f"Feature archive is missing arrays: {sorted(missing_arrays)}"
            )

        y = archive["y"].astype(np.int64, copy=True)
        paths = archive["paths"].astype(str, copy=True)
        class_names = archive["class_names"].astype(str, copy=True)
        person_ids = archive["person_ids"].astype(str, copy=True)
        group_ids = archive["groups"].astype(str, copy=True)

    sample_count = len(y)
    arrays = {
        "paths": paths,
        "person_ids": person_ids,
        "groups": group_ids,
    }

    for name, array in arrays.items():
        if len(array) != sample_count:
            raise ValueError(
                f"Array {name!r} has {len(array)} rows; expected {sample_count}"
            )

    if sample_count == 0:
        raise ValueError("Feature archive contains no samples")

    if len(class_names) == 0:
        raise ValueError("Feature archive contains no class names")

    if np.any(y < 0) or np.any(y >= len(class_names)):
        raise ValueError("The y array contains labels outside class_names")

    if np.any(np.char.strip(person_ids) == ""):
        raise ValueError("person_ids contains empty values")

    if np.any(np.char.strip(group_ids) == ""):
        raise ValueError("groups contains empty values")

    return y, paths, class_names, person_ids, group_ids


def make_recording_groups(
    person_ids: np.ndarray,
    group_ids: np.ndarray,
) -> np.ndarray:
    """Make group IDs globally unique without changing the source metadata."""
    return np.asarray(
        [f"person={person}|group={group}" for person, group in zip(person_ids, group_ids)],
        dtype=str,
    )


def create_split_indices(
    y: np.ndarray,
    recording_groups: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Balance each class while assigning whole recording groups."""
    rng = np.random.default_rng(seed)
    split_indices: dict[str, list[int]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    # In this dataset, each recording group belongs to exactly one pose class.
    # Verify that assumption before splitting class-by-class.
    labels_by_group: dict[str, set[int]] = {}
    for label, group in zip(y, recording_groups):
        labels_by_group.setdefault(str(group), set()).add(int(label))

    mixed_groups = [
        group for group, labels in labels_by_group.items() if len(labels) > 1
    ]
    if mixed_groups:
        raise ValueError(
            "Some recording groups contain multiple classes, so the current "
            "class-by-class group allocator cannot be used. Examples: "
            f"{mixed_groups[:5]}"
        )

    for class_index in range(int(np.max(y)) + 1):
        class_sample_indices = np.flatnonzero(y == class_index)
        class_groups = np.unique(recording_groups[class_sample_indices])
        rng.shuffle(class_groups)

        indices_by_group = {
            str(group): class_sample_indices[
                recording_groups[class_sample_indices] == group
            ]
            for group in class_groups
        }
        group_names = [str(group) for group in class_groups]
        group_sizes = [len(indices_by_group[group]) for group in group_names]

        class_total = len(class_sample_indices)
        validation_target = round(class_total * 0.15)
        test_target = round(class_total * 0.15)
        train_target = class_total - validation_target - test_target

        # Dynamic programming tries every achievable pair of validation/test
        # sample counts. Each tuple records 0=train, 1=validation, 2=test for
        # the groups processed so far.
        states: dict[tuple[int, int], tuple[int, ...]] = {(0, 0): ()}

        for size in group_sizes:
            next_states: dict[tuple[int, int], tuple[int, ...]] = {}

            for (validation_count, test_count), assignments in states.items():
                candidates = (
                    ((validation_count, test_count), assignments + (0,)),
                    ((validation_count + size, test_count), assignments + (1,)),
                    ((validation_count, test_count + size), assignments + (2,)),
                )

                for state, candidate_assignments in candidates:
                    # Keeping the first result provides deterministic tie-breaking.
                    next_states.setdefault(state, candidate_assignments)

            states = next_states

        def state_score(state: tuple[int, int]) -> tuple[int, int, int, int]:
            validation_count, test_count = state
            train_count = class_total - validation_count - test_count

            # Reject states that leave a split empty for this class.
            if min(train_count, validation_count, test_count) <= 0:
                return (10**9, 10**9, validation_count, test_count)

            total_error = (
                abs(train_count - train_target)
                + abs(validation_count - validation_target)
                + abs(test_count - test_target)
            )
            validation_test_difference = abs(validation_count - test_count)
            return (
                total_error,
                validation_test_difference,
                validation_count,
                test_count,
            )

        best_state = min(states, key=state_score)
        assignments = states[best_state]

        split_name_for_assignment = {
            0: "train",
            1: "validation",
            2: "test",
        }
        for group, assignment in zip(group_names, assignments):
            split_name = split_name_for_assignment[assignment]
            split_indices[split_name].extend(indices_by_group[group].tolist())

    return (
        np.sort(np.asarray(split_indices["train"], dtype=np.int64)),
        np.sort(np.asarray(split_indices["validation"], dtype=np.int64)),
        np.sort(np.asarray(split_indices["test"], dtype=np.int64)),
    )


def validate_splits(
    y: np.ndarray,
    recording_groups: np.ndarray,
    class_names: np.ndarray,
    splits: dict[str, np.ndarray],
) -> None:
    all_indices = np.concatenate(list(splits.values()))

    if len(all_indices) != len(y):
        raise RuntimeError(
            f"Splits contain {len(all_indices)} indices for {len(y)} samples"
        )

    unique_indices = np.unique(all_indices)
    if len(unique_indices) != len(y):
        raise RuntimeError("Some samples appear in multiple splits")

    if not np.array_equal(unique_indices, np.arange(len(y))):
        raise RuntimeError("Some dataset samples are missing from the splits")

    split_group_sets = {
        name: set(recording_groups[indices].tolist())
        for name, indices in splits.items()
    }

    split_names = list(splits)
    for first_position, first_name in enumerate(split_names):
        for second_name in split_names[first_position + 1 :]:
            overlap = split_group_sets[first_name] & split_group_sets[second_name]
            if overlap:
                raise RuntimeError(
                    f"Recording-group leakage between {first_name} and "
                    f"{second_name}: {sorted(overlap)[:5]}"
                )

    expected_classes = set(range(len(class_names)))
    for split_name, indices in splits.items():
        present_classes = set(np.unique(y[indices]).tolist())
        missing_classes = expected_classes - present_classes
        if missing_classes:
            missing_names = [class_names[index] for index in sorted(missing_classes)]
            raise RuntimeError(
                f"{split_name} split is missing classes: {missing_names}"
            )


def save_indices(output_directory: Path, splits: dict[str, np.ndarray]) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)

    for split_name, indices in splits.items():
        np.save(output_directory / f"{split_name}_indices.npy", indices)


def save_assignments(
    output_path: Path,
    paths: np.ndarray,
    y: np.ndarray,
    class_names: np.ndarray,
    person_ids: np.ndarray,
    group_ids: np.ndarray,
    recording_groups: np.ndarray,
    splits: dict[str, np.ndarray],
) -> None:
    split_for_index: dict[int, str] = {}
    for split_name, indices in splits.items():
        for index in indices:
            split_for_index[int(index)] = split_name

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "sample_index",
                "split",
                "image_path",
                "class_name",
                "person_id",
                "group_id",
                "recording_group",
            ],
        )
        writer.writeheader()

        for index in range(len(y)):
            writer.writerow(
                {
                    "sample_index": index,
                    "split": split_for_index[index],
                    "image_path": paths[index],
                    "class_name": class_names[y[index]],
                    "person_id": person_ids[index],
                    "group_id": group_ids[index],
                    "recording_group": recording_groups[index],
                }
            )


def print_summary(
    y: np.ndarray,
    class_names: np.ndarray,
    recording_groups: np.ndarray,
    splits: dict[str, np.ndarray],
) -> None:
    total = len(y)
    print("Split summary:")

    for split_name, indices in splits.items():
        percentage = 100.0 * len(indices) / total
        group_count = len(np.unique(recording_groups[indices]))
        print(
            f"\n{split_name}: {len(indices)} samples "
            f"({percentage:.1f}%), {group_count} recording groups"
        )

        counts = np.bincount(y[indices], minlength=len(class_names))
        for class_index, class_name in enumerate(class_names):
            print(f"  {class_name}: {counts[class_index]}")


def main() -> None:
    arguments = parse_arguments()
    project_root = arguments.project_root.expanduser().resolve()
    feature_path = project_root / "features" / "features.npz"
    output_directory = project_root / "splits"

    y, paths, class_names, person_ids, group_ids = load_feature_metadata(feature_path)
    recording_groups = make_recording_groups(person_ids, group_ids)

    train_indices, validation_indices, test_indices = create_split_indices(
        y,
        recording_groups,
        arguments.seed,
    )

    splits = {
        "train": train_indices,
        "validation": validation_indices,
        "test": test_indices,
    }

    validate_splits(y, recording_groups, class_names, splits)
    save_indices(output_directory, splits)
    save_assignments(
        output_directory / "split_assignments.csv",
        paths,
        y,
        class_names,
        person_ids,
        group_ids,
        recording_groups,
        splits,
    )

    print(f"Created split files in: {output_directory}")
    print(f"Random seed: {arguments.seed}")
    print(f"Total samples: {len(y)}")
    print(f"Unique recording groups: {len(np.unique(recording_groups))}")
    print_summary(y, class_names, recording_groups, splits)


if __name__ == "__main__":
    main()
