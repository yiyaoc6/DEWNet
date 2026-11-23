import argparse
import json
from pathlib import Path

import numpy as np

from data_process.my_dataset import aligned_files, CROP_START, CROP_END, files_by_stem, patient_ids, read_mask


def audit(root):
    patients = patient_ids(root)
    if not patients:
        raise ValueError("No patient folders containing image slices were found.")
    total_pixels = outside_pixels = affected_slices = affected_patients = slices = 0
    for patient in patients:
        directory = root / patient
        labels = aligned_files(directory, "label", files_by_stem(directory / "image"))
        had_loss = False
        for path in labels.values():
            foreground = np.asarray(read_mask(path)) != 0
            size = np.count_nonzero(foreground)
            cropped = np.count_nonzero(foreground[CROP_START:CROP_END, CROP_START:CROP_END])
            lost = int(size - cropped)
            slices += 1
            total_pixels += int(size)
            outside_pixels += lost
            affected_slices += bool(lost)
            had_loss |= bool(lost)
        affected_patients += had_loss
    return {"patients": len(patients), "slices": slices, "foreground_pixels": total_pixels,
            "foreground_pixels_outside_crop": outside_pixels, "affected_slices": affected_slices,
            "affected_patients": affected_patients}


def main():
    parser = argparse.ArgumentParser(description="Count annotated PE pixels outside the central crop.")
    parser.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.data_root)
    print(json.dumps(report, indent=2))
    if report["foreground_pixels_outside_crop"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
