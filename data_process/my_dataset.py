from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset


ORIGINAL_SIZE = 512
CROP_SIZE = 448
CROP_START = (ORIGINAL_SIZE - CROP_SIZE) // 2
CROP_END = CROP_START + CROP_SIZE
CROP_BOX = (CROP_START, CROP_START, CROP_END, CROP_END)
EXTENSIONS = {".bmp", ".png", ".tif", ".tiff", ".jpg", ".jpeg"}


def patient_ids(root):
    return sorted(
        path.name for path in Path(root).iterdir()
        if path.is_dir() and not path.name.startswith(".") and (path / "image").is_dir()
    )


def files_by_stem(directory):
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing slice directory: {directory}")
    files = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in EXTENSIONS and not path.name.startswith("."):
            if path.stem in files:
                raise ValueError(f"Duplicate slice stem {path.stem!r} in {directory}")
            files[path.stem] = path
    if not files:
        raise ValueError(f"No image slices found in {directory}")
    return files


def aligned_files(patient_dir, folder, images):
    files = files_by_stem(patient_dir / folder)
    if files.keys() != images.keys():
        missing = sorted(images.keys() - files.keys())
        extra = sorted(files.keys() - images.keys())
        raise ValueError(f"Slice stems differ in {patient_dir / folder}: missing={missing[:5]}, extra={extra[:5]}")
    return files


def read_gray(path):
    with Image.open(path) as source:
        image = source.convert("L")
    if image.size != (ORIGINAL_SIZE, ORIGINAL_SIZE):
        raise ValueError(f"Expected {ORIGINAL_SIZE}×{ORIGINAL_SIZE} pixels: {path} has {image.size}")
    return image


def read_mask(path):
    image = read_gray(path)
    values = set(np.unique(np.asarray(image)))
    if not (values <= {0, 1} or values <= {0, 255}):
        raise ValueError(f"Expected binary labels encoded as 0/1 or 0/255 in {path}")
    return image.point(lambda value: 255 if value else 0)


class LungPEDataset(Dataset):
    def __init__(self, root, patients, *, labels=True, augment=False, apply_roi=False):
        self.items = []
        self.labels = labels
        self.augment = augment
        for patient_id in patients:
            directory = Path(root) / patient_id
            images = files_by_stem(directory / "image")
            masks = aligned_files(directory, "label", images) if labels else {}
            rois = aligned_files(directory, "mask", images) if apply_roi else {}
            self.items.extend(
                (patient_id, stem, image, masks.get(stem), rois.get(stem))
                for stem, image in images.items()
            )
        if not self.items:
            raise ValueError("The selected patient set contains no image slices.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        patient, stem, image_path, label_path, roi_path = self.items[index]
        image = read_gray(image_path)
        if roi_path is not None:
            roi = np.asarray(read_mask(roi_path)) != 0
            image = Image.fromarray(np.where(roi, np.asarray(image), 0).astype(np.uint8))
        image = image.crop(CROP_BOX)
        if self.labels:
            label = read_mask(label_path).crop(CROP_BOX)
            if self.augment:
                angle = random.uniform(-15, 15)
                image = image.rotate(angle, resample=Image.Resampling.BILINEAR)
                label = label.rotate(angle, resample=Image.Resampling.NEAREST)
                if random.random() < 0.5:
                    image, label = ImageOps.mirror(image), ImageOps.mirror(label)
                if random.random() < 0.5:
                    image, label = ImageOps.flip(image), ImageOps.flip(label)
            target = torch.from_numpy((np.asarray(label) != 0).astype(np.int64))
        tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).unsqueeze(0)
        return (tensor, target) if self.labels else (tensor, patient, stem)


def restore_mask(crop):
    if crop.shape != (CROP_SIZE, CROP_SIZE):
        raise ValueError(f"Expected a {CROP_SIZE}×{CROP_SIZE} prediction, got {crop.shape}")
    result = np.zeros((ORIGINAL_SIZE, ORIGINAL_SIZE), dtype=np.uint8)
    result[CROP_START:CROP_END, CROP_START:CROP_END] = crop.astype(np.uint8) * 255
    return result
