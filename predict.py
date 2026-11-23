import argparse
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader

from data_process.my_dataset import LungPEDataset, patient_ids, restore_mask
from data_process.split_dataset import GROUPS, load_split
from models import DEWNet, VARIANTS
from utils.train_and_eval import foreground_mask


def load_model(checkpoint_path, variant, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" in checkpoint:
        if variant is not None and variant != checkpoint["variant"]:
            raise ValueError("Requested variant disagrees with the checkpoint metadata.")
        variant = checkpoint["variant"]
        checkpoint = checkpoint["model"]
    model = DEWNet(variant or "full")
    model.load_state_dict(checkpoint)
    return model.to(device).eval()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Predict binary 512×512 PE masks.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--group", choices=GROUPS, default="test")
    parser.add_argument("--variant", choices=VARIANTS, help="Variant for a bare model state dict.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--apply-roi", action="store_true", help="Use the same input masking as training.")
    args = parser.parse_args()
    if args.fold is not None and args.split_file is None:
        parser.error("--fold requires --split-file.")

    device = torch.device(args.device)
    model = load_model(args.checkpoint, args.variant, device)
    patients = patient_ids(args.data_root)
    if args.split_file:
        patients = load_split(args.split_file, patients, args.fold)[args.group]
    if args.output.resolve() == args.data_root.resolve():
        raise ValueError("Use a separate directory for predictions.")
    dataset = LungPEDataset(args.data_root, patients, labels=False, apply_roi=args.apply_roi)
    loader = DataLoader(dataset, batch_size=args.batch_size)

    for images, patients, stems in loader:
        masks = foreground_mask(model(images.to(device))).cpu().numpy()
        for mask, patient, stem in zip(masks, patients, stems):
            directory = args.output / patient
            directory.mkdir(parents=True, exist_ok=True)
            Image.fromarray(restore_mask(mask)).save(directory / f"{stem}.png")
    print(f"Saved {len(dataset)} prediction masks to {args.output}.")


if __name__ == "__main__":
    main()
