from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from io import TextIOWrapper
from pathlib import Path
from types import ModuleType

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover (Windows)
    _fcntl = None

fcntl_module: ModuleType | None = _fcntl


DEFAULT_BITS = [24, 16, 12, 10, 8, 6, 4, 2]
DEFAULT_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
FIXED_DAC_BITS = 4
FIXED_ADC_BITS = 6


@dataclass(frozen=True)
class RunSpec:
    bits: int
    gpu: int


@dataclass
class RunHandle:
    spec: RunSpec
    output_dir: Path
    log_path: Path
    start_time_s: float
    proc: subprocess.Popen[str]
    log_file: TextIOWrapper


def _parse_int_list(arg: str) -> list[int]:
    vals: list[int] = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    return vals


def _read_metrics_summary(output_dir: Path) -> dict[str, object]:
    summary_path = output_dir / "metrics_summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _append_row_locked(
    csv_path: Path, row: dict[str, object], fieldnames: list[str]
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a+", newline="", encoding="utf-8") as f:
        if fcntl_module is not None:
            fcntl_module.flock(f.fileno(), fcntl_module.LOCK_EX)
        f.seek(0, os.SEEK_END)
        is_empty = f.tell() == 0
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_empty:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())
        if fcntl_module is not None:
            fcntl_module.flock(f.fileno(), fcntl_module.LOCK_UN)


def _launch_run(
    repo_root: Path,
    cfg_path: Path,
    output_root: Path,
    bits: int,
    gpu: int,
    batch_size: int,
    num_epochs: int | None,
    max_train_batches: int | None,
    max_eval_batches: int | None,
    conv: str,
) -> RunHandle:
    spec = RunSpec(bits=bits, gpu=gpu)
    output_dir = output_root / f"bits_{bits}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"

    cmd = [
        sys.executable,
        str(repo_root / "onn_main.py"),
        "--config-file",
        str(cfg_path),
        "--output-dir",
        str(output_dir),
        "--conv-backend",
        conv,
        "--batch-size",
        str(batch_size),
        "--quantizer",
        "ste_maxscale",
        "--dac-bits",
        str(FIXED_DAC_BITS),
        "--adc-bits",
        str(FIXED_ADC_BITS),
        "--fourier-plane-bits",
        str(bits),
        "--run-pretrain-tests",
        "false",
    ]
    if num_epochs is not None:
        cmd += ["--num-epochs", str(num_epochs)]
    if max_train_batches is not None:
        cmd += ["--max-train-batches", str(max_train_batches)]
    if max_eval_batches is not None:
        cmd += ["--max-eval-batches", str(max_eval_batches)]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTHONUNBUFFERED", "1")

    log_fh = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    return RunHandle(
        spec=spec,
        output_dir=output_dir,
        log_path=log_path,
        start_time_s=time.time(),
        proc=proc,
        log_file=log_fh,
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Parallel sweep over quantization bit-widths using ste_maxscale."
    )
    parser.add_argument(
        "--bits",
        type=str,
        default=",".join(str(x) for x in DEFAULT_BITS),
        help="Comma-separated list of bit-widths to run.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=",".join(str(x) for x in DEFAULT_GPUS),
        help="Comma-separated GPU ids to use (1 run per GPU).",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=str(repo_root / "configs" / "config_ideal.yaml"),
        help="Base YAML config file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Training batch size.",
    )
    parser.add_argument(
        "--conv",
        type=str,
        choices=["fourier", "pytorch"],
        default="fourier",
        help="Convolution backend to use.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Override number of epochs (default: config).",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Debug: limit training batches per epoch for launched runs.",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        help="Debug: limit eval batches for launched runs.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="",
        help="Where to write per-run outputs (default: runs/quantlevel_ste_maxscale_<timestamp>).",
    )
    parser.add_argument(
        "--shared-csv",
        type=str,
        default=str(repo_root / "sweep_results" / "quantlevel_ste_maxscale.csv"),
        help="Shared CSV path for best/final metrics across runs.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=30.0,
        help="Polling interval for checking child process completion.",
    )
    args = parser.parse_args()

    bits_list = _parse_int_list(args.bits)
    gpus_list = _parse_int_list(args.gpus)
    if not bits_list:
        raise ValueError("No --bits provided.")
    if not gpus_list:
        raise ValueError("No --gpus provided.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_root:
        output_root = Path(args.output_root)
    else:
        output_root = repo_root / "runs" / f"quantlevel_ste_maxscale_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(args.config_file)
    if not cfg_path.is_absolute():
        cfg_path = repo_root / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    shared_csv = Path(args.shared_csv)
    if not shared_csv.is_absolute():
        shared_csv = repo_root / shared_csv

    shared_fields = [
        "bits",
        "gpu",
        "output_dir",
        "returncode",
        "best_test_epoch",
        "best_test_acc",
        "final_epoch",
        "final_test_acc",
        "duration_sec",
    ]

    # Prevent silently corrupting an existing CSV with a different header schema.
    if shared_csv.exists():
        with shared_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                existing_header = next(reader)
            except StopIteration:
                existing_header = []
        existing_header = [h.strip() for h in existing_header if h is not None]
        if existing_header and existing_header != shared_fields:
            raise ValueError(
                f"Shared CSV header mismatch for {shared_csv}.\n"
                f"  existing: {existing_header}\n"
                f"  expected: {shared_fields}\n"
                "Use --shared-csv to choose a new output file (or delete the existing CSV)."
            )

    pending_bits = list(bits_list)
    available_gpus = list(gpus_list)
    running: list[RunHandle] = []

    print(f"[INFO] Repo root: {repo_root}")
    print(f"[INFO] Output root: {output_root}")
    print(f"[INFO] Shared CSV: {shared_csv}")
    print(f"[INFO] Bits: {bits_list}")
    print(f"[INFO] GPUs: {gpus_list}")

    while pending_bits or running:
        while pending_bits and available_gpus:
            bits = pending_bits.pop(0)
            gpu = available_gpus.pop(0)
            handle = _launch_run(
                repo_root=repo_root,
                cfg_path=cfg_path,
                output_root=output_root,
                bits=bits,
                gpu=gpu,
                batch_size=int(args.batch_size),
                num_epochs=args.num_epochs,
                max_train_batches=args.max_train_batches,
                max_eval_batches=args.max_eval_batches,
                conv=str(args.conv),
            )
            running.append(handle)
            print(
                f"[LAUNCH] bits={bits} gpu={gpu} -> {handle.output_dir} (log: {handle.log_path})"
            )

        finished: list[RunHandle] = []
        for handle in running:
            rc = handle.proc.poll()
            if rc is None:
                continue

            duration_s = time.time() - handle.start_time_s
            handle.log_file.close()

            summary = _read_metrics_summary(handle.output_dir)
            best_test = (
                summary.get("best_test", {}) if isinstance(summary, dict) else {}
            )
            final = summary.get("final", {}) if isinstance(summary, dict) else {}

            row = {
                "bits": handle.spec.bits,
                "gpu": handle.spec.gpu,
                "output_dir": str(handle.output_dir),
                "returncode": rc,
                "best_test_epoch": best_test.get("epoch", ""),
                "best_test_acc": best_test.get("test_acc", ""),
                "final_epoch": final.get("epoch", ""),
                "final_test_acc": final.get("test_acc", ""),
                "duration_sec": f"{duration_s:.1f}",
            }
            _append_row_locked(shared_csv, row, shared_fields)
            print(f"[DONE] bits={handle.spec.bits} gpu={handle.spec.gpu} rc={rc}")

            available_gpus.append(handle.spec.gpu)
            finished.append(handle)

        for h in finished:
            running.remove(h)

        if running:
            time.sleep(float(args.poll_seconds))

    print(f"[INFO] Sweep complete. Results appended to {shared_csv}")


if __name__ == "__main__":
    main()
