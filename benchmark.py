import argparse
import json
from statistics import mean, stdev

import torch

from data_process.my_dataset import CROP_SIZE
from models import DEWNet, VARIANTS


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Benchmark CUDA inference at FP32, batch size one.")
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--passes", type=int, default=100, help="Forward passes in each timed round.")
    args = parser.parse_args()
    if args.rounds < 1 or args.passes < 1 or args.warmup < 0:
        parser.error("Provide positive rounds and passes, and nonnegative warm-up passes.")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("This benchmark requires CUDA.")

    with torch.cuda.device(device):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        model = DEWNet(args.variant).to(device).eval()
        image = torch.randn(1, 1, CROP_SIZE, CROP_SIZE, device=device)
        for _ in range(args.warmup):
            model(image)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(args.rounds):
            start.record()
            for _ in range(args.passes):
                model(image)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) / args.passes)
        print(json.dumps({
            "variant": args.variant, "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "warmup": args.warmup, "passes": args.passes,
            "rounds_ms_per_slice": times, "mean_ms_per_slice": mean(times),
            "sample_sd_ms_per_slice": stdev(times) if len(times) > 1 else None,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024 ** 2,
        }, indent=2))


if __name__ == "__main__":
    main()
