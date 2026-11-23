import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from data_process.my_dataset import aligned_files, files_by_stem, patient_ids, read_mask
from data_process.split_dataset import GROUPS, load_split
from utils.metrics import BinaryMetrics


def evaluate(data_root, prediction_root, patients):
    if not patients:
        raise ValueError("The selected set contains no patients.")
    overall = BinaryMetrics()
    by_patient = {}
    for patient in patients:
        patient_dir = data_root / patient
        images = files_by_stem(patient_dir / "image")
        labels = aligned_files(patient_dir, "label", images)
        predictions = files_by_stem(prediction_root / patient)
        if predictions.keys() != labels.keys():
            missing = sorted(labels.keys() - predictions.keys())
            extra = sorted(predictions.keys() - labels.keys())
            raise ValueError(f"Prediction slices differ for {patient}: missing={missing[:5]}, extra={extra[:5]}")
        stats = BinaryMetrics()
        for stem, path in labels.items():
            label = np.asarray(read_mask(path)) != 0
            mask = np.asarray(read_mask(predictions[stem])) != 0
            stats.update(mask, label)
        by_patient[patient] = {"counts": stats.counts(), **stats.scores()}
        overall.merge(stats)
    overall_scores = overall.scores()
    summary = {}
    for metric in overall_scores:
        values = [patient[metric] for patient in by_patient.values()]
        summary[metric] = {"mean": mean(values), "sample_sd": stdev(values) if len(values) > 1 else None}
    return {"overall": {"counts": overall.counts(), **overall_scores},
            "patient_summary": {"patients": len(by_patient), **summary},
            "per_patient": by_patient}


def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation masks at slice and patient level.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--group", choices=GROUPS, default="test")
    parser.add_argument("--output", type=Path, help="Write JSON including per-patient scores.")
    args = parser.parse_args()
    if args.fold is not None and args.split_file is None:
        parser.error("--fold requires --split-file.")
    selected = patient_ids(args.data_root)
    if args.split_file:
        selected = load_split(args.split_file, selected, args.fold)[args.group]
    report = evaluate(args.data_root, args.predictions, selected)
    print(json.dumps({key: report[key] for key in ("overall", "patient_summary")}, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
