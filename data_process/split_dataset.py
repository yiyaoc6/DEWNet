import argparse
import json
from pathlib import Path
import random

from .my_dataset import patient_ids


GROUPS = ("train", "val", "test")


def validate_split(split, available):
    if not isinstance(split, dict) or set(split) != set(GROUPS):
        raise ValueError("Each split requires exactly train, val, and test lists.")
    seen = set()
    available = set(available)
    for group in GROUPS:
        ids = split[group]
        if not isinstance(ids, list) or not ids or not all(isinstance(patient, str) for patient in ids):
            raise ValueError(f"{group} must be a non-empty list of patient IDs.")
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError(f"Duplicate patient within or between partitions: {group}")
        if not set(ids) <= available:
            raise ValueError(f"Ground-truth data root is missing patients in {group}.")
        seen.update(ids)
    return split


def load_split(path, available, fold=None):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "folds" in data:
        folds = data["folds"]
        if fold is None or not 1 <= fold <= len(folds):
            raise ValueError(f"Select a fold between 1 and {len(folds)} with --fold.")
        validated = [validate_split(split, available) for split in folds]
        population = set.union(*(set(part) for part in validated[0].values()))
        held_out = [patient for split in validated for patient in split["test"]]
        if len(held_out) != len(set(held_out)) or set(held_out) != population:
            raise ValueError("The fold test sets must partition the selected patients exactly once.")
        if any(set.union(*(set(part) for part in split.values())) != population for split in validated):
            raise ValueError("All folds must use the same patient cohort.")
        return validated[fold - 1]
    if fold is not None:
        raise ValueError("--fold requires a cross-validation manifest.")
    return validate_split(data, available)


def make_holdout(ids, seed):
    if len(ids) < 3:
        raise ValueError("A holdout split requires at least three patients.")
    rng = random.Random(seed)
    ids = sorted(ids)
    rng.shuffle(ids)
    train = min(max(1, int(len(ids) * 0.7)), len(ids) - 2)
    val = max(1, int(len(ids) * 0.1))
    return {"train": sorted(ids[:train]), "val": sorted(ids[train:train + val]),
            "test": sorted(ids[train + val:])}


def make_folds(ids, seed, validation_patients=8):
    if len(ids) < 5 or not 0 < validation_patients < len(ids) - (len(ids) + 4) // 5:
        raise ValueError("Each fold needs test, validation, and training patients.")
    rng = random.Random(seed)
    ids = sorted(ids)
    rng.shuffle(ids)
    size, larger_folds = divmod(len(ids), 5)
    folds = []
    start = 0
    for index in range(5):
        end = start + size + (index < larger_folds)
        test = ids[start:end]
        candidates = ids[:start] + ids[end:]
        rng.shuffle(candidates)
        folds.append({"train": sorted(candidates[validation_patients:]),
                      "val": sorted(candidates[:validation_patients]), "test": sorted(test)})
        start = end
    return {"folds": folds}


def main():
    parser = argparse.ArgumentParser(description="Generate a patient-level experiment manifest.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("holdout", "fivefold"), default="holdout")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-patients", type=int, default=8, help="Five-fold validation count.")
    args = parser.parse_args()
    ids = patient_ids(args.data_root)
    manifest = (make_holdout(ids, args.seed) if args.mode == "holdout"
                else make_folds(ids, args.seed, args.validation_patients))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
