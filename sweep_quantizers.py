import os
import re
import subprocess
import sys
from typing import List, Tuple


QUANTIZERS: List[str] = [
    "ste_clipped",
    "ste_maxscale",
    "ios",
    "mad",
    "mph",
    "pwl",
]


def run_once(quantizer: str) -> Tuple[str, float | None, int, str]:
    """Run onn_main.py with fixed args and given quantizer.

    Returns: (quantizer, best_acc, returncode, stdout)
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(project_root, "runs", "runs_pytorch", quantizer)

    cmd = [
        sys.executable,
        os.path.join(project_root, "onn_main.py"),
        "--config-file",
        os.path.join(project_root, "configs", "config_ideal.yaml"),
        "--output-dir",
        output_dir,
        "--use-fourier-conv",
        "--quantize-fourier-plane",
        "--fourier-plane-bits",
        "4",
        "--batch-size",
        "128",
        "--quantizer",
        quantizer,
    ]

    proc = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
    stdout = proc.stdout + "\n" + proc.stderr

    # Parse last-epoch test accuracy from the final training summary line
    # Example: "Training finished. Last epoch test accuracy: 76.34% | Best test accuracy: ..."
    last_acc = None
    m = re.search(r"Last epoch test accuracy:\s*([0-9]+\.?[0-9]*)%", stdout)
    if m:
        try:
            last_acc = float(m.group(1))
        except ValueError:
            last_acc = None
    else:
        # Fallback: take the last printed "Accuracy: xx.xx%" from evaluate
        matches = re.findall(r"Accuracy:\s*([0-9]+\.?[0-9]*)%", stdout)
        if matches:
            try:
                last_acc = float(matches[-1])
            except ValueError:
                last_acc = None

    return quantizer, last_acc, proc.returncode, stdout


def main() -> None:
    results: List[Tuple[str, float | None, int]] = []
    print("Starting quantizer sweep...\n")
    for q in QUANTIZERS:
        print(f"=== Running quantizer: {q} ===")
        quant, acc, code, _ = run_once(q)
        results.append((quant, acc, code))
        status = "OK" if code == 0 else f"RC={code}"
        acc_str = f"{acc:.2f}%" if acc is not None else "N/A"
        print(f" -> {quant}: {acc_str} ({status})\n")

    print("Summary (last-epoch test accuracy):")
    for quant, acc, code in results:
        acc_str = f"{acc:.2f}%" if acc is not None else "N/A"
        status = "OK" if code == 0 else f"RC={code}"
        print(f" - {quant:>13}: {acc_str}  [{status}]")


if __name__ == "__main__":
    main()


