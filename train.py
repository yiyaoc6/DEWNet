import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_process.my_dataset import LungPEDataset, patient_ids
from data_process.split_dataset import load_split
from models import DEWNet, VARIANTS
from utils.train_and_eval import (
    create_lr_scheduler, seed_everything, seed_worker, train_one_epoch, validation_f1,
)


def save_checkpoint(path, checkpoint):
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def write_history(path, history):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def random_state(train_generator, val_generator, device):
    kind, keys, position, has_gauss, gaussian = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (kind, keys.tolist(), position, has_gauss, gaussian),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        "train": train_generator.get_state(),
        "validation": val_generator.get_state(),
    }


def restore_random_state(state, train_generator, val_generator, device):
    random.setstate(state["python"])
    kind, keys, position, has_gauss, gaussian = state["numpy"]
    np.random.set_state((kind, np.asarray(keys, dtype=np.uint32), position, has_gauss, gaussian))
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)
    train_generator.set_state(state["train"])
    val_generator.set_state(state["validation"])


def main():
    parser = argparse.ArgumentParser(description="Train DEWNet with patient-level validation.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--apply-roi", action="store_true", help="Optional input masking for legacy data.")
    parser.add_argument("--resume", type=Path, help="Resume from a compatible last.pt.")
    args = parser.parse_args()

    device = torch.device(args.device)
    seed_everything(args.seed)
    split = load_split(args.split_file, patient_ids(args.data_root), args.fold)
    train_data = LungPEDataset(args.data_root, split["train"], augment=True, apply_roi=args.apply_roi)
    val_data = LungPEDataset(args.data_root, split["val"], apply_roi=args.apply_roi)
    train_generator = torch.Generator().manual_seed(args.seed)
    val_generator = torch.Generator().manual_seed(args.seed)
    loader_options = {"num_workers": args.num_workers, "worker_init_fn": seed_worker,
                      "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              generator=train_generator, **loader_options)
    val_loader = DataLoader(val_data, batch_size=1, generator=val_generator, **loader_options)

    model = DEWNet(args.variant).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, args.warmup_epochs)
    config = {
        "variant": args.variant, "seed": args.seed, "batch_size": args.batch_size,
        "epochs": args.epochs, "warmup_epochs": args.warmup_epochs,
        "steps_per_epoch": len(train_loader), "num_workers": args.num_workers,
        "apply_roi": args.apply_roi, "device_type": device.type,
        "split_sha256": hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest(),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    best_path = args.output / "best.pt"
    last_path = args.output / "last.pt"
    history_path = args.output / "history.json"
    start_epoch, best_metric, best_epoch, history = 0, -1.0, 0, []
    best_state = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint["config"] != config:
            raise ValueError("Resume configuration differs from the saved training configuration.")
        if args.resume.resolve() != last_path.resolve():
            if any(path.exists() for path in (best_path, last_path, history_path)):
                raise FileExistsError(f"Resume output already has a run: {args.output}")
            save_checkpoint(last_path, checkpoint)
        best_state = checkpoint["best_model"]
        save_checkpoint(best_path, {"model": best_state, "variant": args.variant,
                                    "epoch": checkpoint["best_epoch"],
                                    "validation_f1": checkpoint["best_f1"]})
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch, best_metric = checkpoint["epoch"], checkpoint["best_f1"]
        best_epoch, history = checkpoint["best_epoch"], checkpoint["history"]
        write_history(history_path, history)
        restore_random_state(checkpoint["random_state"], train_generator, val_generator, device)
    elif any(path.exists() for path in (best_path, last_path, history_path)):
        raise FileExistsError(f"Run already exists in {args.output}. Select a new output or resume.")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_loss = train_one_epoch(model, optimizer, scheduler, train_loader, device)
        f1 = validation_f1(model, val_loader, device)
        entry = {"epoch": epoch, "loss": train_loss, "validation_f1": f1,
                 "learning_rate": scheduler.get_last_lr()[0]}
        history.append(entry)
        improved = f1 > best_metric
        if improved:
            best_metric, best_epoch = f1, epoch
            best_state = {name: value.detach().to("cpu", copy=True)
                          for name, value in model.state_dict().items()}
        save_checkpoint(last_path, {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "variant": args.variant, "config": config,
            "epoch": epoch, "best_f1": best_metric, "best_epoch": best_epoch,
            "best_model": best_state, "history": history,
            "random_state": random_state(train_generator, val_generator, device),
        })
        if improved:
            save_checkpoint(best_path, {"model": best_state, "variant": args.variant,
                                        "epoch": best_epoch, "validation_f1": best_metric})
        write_history(history_path, history)
        print(json.dumps(entry), flush=True)


if __name__ == "__main__":
    main()
