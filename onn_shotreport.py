"""Connect shot accounting and physical assumptions to workload measurements."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from onn_config import load_app_config_from_yaml
from onn_jtc_conv2d import JTCConv2d
from onn_remodulation import AnalogRemodulator, RemodulationSpec


def physical_settings(config):
    """Include transfer, noise, converter and gain settings beside the plan."""
    values = asdict(config)
    names = {
        "conv_backend",
        "dac_bits",
        "adc_bits",
        "fourier_plane_bits",
        "loss",
        "laser_power_gain",
        "laser_rin_db",
        "ler_std_dev",
        "pd_noise_w",
        "tia_input_bias",
        "converter_clamp_grad",
        "scale_output",
        "transfer_linearization",
    }
    return {
        name: value
        for name, value in values.items()
        if name in names
        or name.startswith("jtc_")
        or name.startswith("pd_input_")
        or "distortion" in name
        or name.startswith("mrm_")
        or name.startswith("lens_")
    }


def observed_shot_plans(model):
    """Unique observed shapes per layer, not an execution-frequency total."""
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, JTCConv2d) and module._shot_plans:
            layers.append(
                {
                    "layer": name,
                    "plans": [plan.to_dict() for plan in module._shot_plans.values()],
                    "physical_settings": physical_settings(module.config),
                    "remodulation": module.shot.remodulator.description(),
                }
            )
    return {
        "schema_version": 1,
        "coverage": "unique observed shapes per layer; counts are per invocation, per rank",
        "layers": layers,
    }


def plan_resnet(config, input_shape=(1, 3, 32, 32)):
    """Dry-run native shape propagation; never execute optics or load a dataset.

    The model factory uses the same ResNet graph for native and optical layers.
    Record every Conv2d invocation, including electronic 1x1 bypasses.
    """
    from onn_train import build_model

    if config.model_arch not in {"resnet11", "resnet18"}:
        raise ValueError(
            "Whole-model ShotPlan currently supports resnet11/resnet18; FTconvlayer has a separate tiling contract"
        )
    if config.conv_backend not in {"jtc_analytic", "jtc_analog_fourier"}:
        raise ValueError("Select an analog Fourier-plane backend for ShotPlan")
    if len(input_shape) != 4 or input_shape[1] != 3 or min(input_shape) <= 0:
        raise ValueError("Expected positive BCHW dimensions with three input channels")
    # Counts scale linearly in B, so shape propagation needs only one sample.
    batch = int(input_shape[0])
    calls = []
    handles = []
    with torch.random.fork_rng(devices=[]):
        native = build_model(
            replace(config, conv_backend="pytorch", compile_jtc=False)
        ).eval()
        for name, module in native.named_modules():
            if not isinstance(module, torch.nn.Conv2d):
                continue
            adapter = JTCConv2d.from_conv2d(
                module,
                config=replace(config, compile_jtc=False),
                assume_nonnegative_input=config.jtc_assume_nonnegative_input,
            )
            adapter.shot.validate()

            def capture(layer, args, *, adapter=adapter, name=name):
                shape = (batch, *args[0].shape[1:])
                calls.append({"layer": name, **adapter.shot_plan(shape).to_dict()})

            handles.append(module.register_forward_pre_hook(capture))
        try:
            with torch.no_grad():
                output = native(torch.zeros((1, *input_shape[1:])))
        finally:
            for handle in handles:
                handle.remove()
    return {
        "schema_version": 1,
        "model_arch": config.model_arch,
        "input_shape": tuple(input_shape),
        "output_shape": (batch, *output.shape[1:]),
        "scope": "one forward invocation; no retries or calibration shots",
        "physical_settings": physical_settings(config),
        "remodulation": AnalogRemodulator(
            RemodulationSpec(**config.jtc_remodulation)
        ).description(),
        "layers": calls,
        "total_shots": sum(call["shots"] for call in calls),
        "total_adc_samples": sum(call["adc_samples"] for call in calls),
        "total_used_adc_samples": sum(call["used_adc_samples"] for call in calls),
        "total_dac_samples_without_reuse": sum(
            call["dac_samples_without_reuse"] for call in calls
        ),
        "overlapping_used_adc_samples": sum(
            call["shots"] * call["contaminated_selected_lags"] for call in calls
        ),
        "accuracy_impact": "requires workload evaluation; geometric overlap is not an error magnitude",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export ResNet physical shot counts and overlap without running optics."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--input-shape",
        nargs=4,
        type=int,
        default=(1, 3, 32, 32),
        metavar=("B", "C", "H", "W"),
    )
    parser.add_argument("--mapping", choices=["row", "dot"])
    parser.add_argument("--model-arch", choices=["resnet11", "resnet18"])
    parser.add_argument("--total-field", type=int)
    parser.add_argument("--separation", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_app_config_from_yaml(args.config)
    overrides = {
        name: value
        for name, value in {
            "jtc_shot_mapping": args.mapping,
            "model_arch": args.model_arch,
            "jtc_total_field": args.total_field,
            "jtc_separation": args.separation,
        }.items()
        if value is not None
    }
    if args.total_field is not None or args.separation is not None:
        overrides["jtc_rowwise_geometry"] = "config"
    report = plan_resnet(replace(config, **overrides), args.input_shape)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"{report['total_shots']:,} shots; {report['total_adc_samples']:,} ADC samples; {args.output}"
    )


if __name__ == "__main__":
    main()
