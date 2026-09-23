"""Checkpoint contents and exact single-rank/distributed training replay."""

import random
import shutil
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from analog_cases import analog_config, analog_layer
from torch.utils.data import DataLoader, Dataset

import onn_train as train
from onn_config import AppConfig
from onn_train import (
    DistributedContext,
    evaluate_metrics,
    evaluate_metrics_distributed,
    load_training_checkpoint,
    save_checkpoint,
)
from onn_training_state import EpochDataset, EpochSampler


def test_training_checkpoint_round_trip_restores_training_state(tmp_path):
    torch.manual_seed(11)
    config = AppConfig(run_pretrain_tests=False)
    model = nn.Linear(3, 2)
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    loss = model(torch.randn(4, 3)).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()

    saved_weight = model.weight.detach().clone()
    saved_bias = model.bias.detach().clone()
    ckpt_path = tmp_path / "latest_checkpoint.pth"

    save_checkpoint(
        model,
        config,
        best_acc=37.5,
        filename=str(ckpt_path),
        epoch=3,
        optimizer=optimizer,
        scheduler=scheduler,
        best_epoch=2,
        best_snapshot={"test_acc": 37.5},
        checkpoint_kind="time",
        epoch_complete=False,
        batch_idx=17,
        global_step=123,
        data_seed=11,
        best_model_state_dict={k: v.clone() for k, v in model.state_dict().items()},
    )

    restored = nn.Linear(3, 2)
    restored_optimizer = optim.AdamW(restored.parameters(), lr=0.02)
    restored_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        restored_optimizer,
        T_max=5,
    )

    ckpt = load_training_checkpoint(
        str(ckpt_path),
        restored,
        torch.device("cpu"),
        config=replace(
            config,
            output_dir=str(tmp_path / "resumed"),
            resume_checkpoint=str(ckpt_path),
            checkpoint_interval=3,
            dataloader_num_workers=0,
            show_progress=False,
        ),
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
    )

    torch.testing.assert_close(restored.weight, saved_weight)
    torch.testing.assert_close(restored.bias, saved_bias)
    assert ckpt["epoch"] == 3
    assert ckpt["best_accuracy"] == 37.5
    assert ckpt["best_epoch"] == 2
    assert ckpt["best_snapshot"] == {"test_acc": 37.5}
    assert ckpt["checkpoint_kind"] == "time"
    assert ckpt["epoch_complete"] is False
    assert ckpt["batch_idx"] == 17
    assert ckpt["global_step"] == 123
    assert restored_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    assert restored_scheduler.last_epoch == scheduler.last_epoch


def test_distributed_eval_helper_matches_local_eval_when_single_rank():
    model = nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        model.bias.zero_()
    inputs = torch.tensor([[2.0, 1.0], [0.5, 3.0], [4.0, 1.0]])
    labels = torch.tensor([0, 1, 1])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs, labels),
        batch_size=2,
    )
    criterion = nn.CrossEntropyLoss()

    local_loss, local_acc = evaluate_metrics(
        model,
        loader,
        torch.device("cpu"),
        criterion,
    )
    dist_loss, dist_acc = evaluate_metrics_distributed(
        model,
        loader,
        torch.device("cpu"),
        criterion,
        DistributedContext(),
    )

    assert dist_loss == local_loss
    assert dist_acc == local_acc


class AugmentedData(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        x = torch.full((2, 6, 6), (index + 1) / 10)
        x += (random.random() + np.random.random() + torch.rand(())) * 0.02
        return x, index % 2


class ReplayModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.optical = analog_layer(config)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(108, 2))
        if dist.is_initialized():
            torch.rand(dist.get_rank())  # ranks must retain distinct noise streams

    def forward(self, x):
        return self.head(self.optical(x))


def replay_loaders(
    batch_size, *, seed, num_workers, distributed=False, rank=0, world_size=1, **kwargs
):
    dataset = EpochDataset(AugmentedData(), seed)
    sampler = EpochSampler(
        dataset, num_replicas=world_size if distributed else 1, rank=rank, seed=seed
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )
    return loader, [], []


def fake_evaluate(*args, **kwargs):
    # Evaluation must not shift the stochastic training trajectory.
    random.random()
    np.random.random()
    torch.randn(11)
    return 0.0, 100.0


def assert_equal_state(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_equal_state(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_equal_state(a, b)
    else:
        assert actual == expected


def _exercise_replay(tmp_path, workers, rank=0, world_size=1):
    cfg = analog_config(
        jtc_output_gain_mode="calibrate_freeze",
        jtc_gain_freeze_batches=1,
        jtc_gain_recal_epochs=1,
        jtc_frontend_snr_db=35.0,
        seed=17,
        num_epochs=2,
        batch_size=2,
        run_pretrain_tests=False,
        run_full_strength_inference=False,
        show_progress=False,
        train_eval_interval=0,
        output_dir=str(tmp_path / "original"),
        dataloader_num_workers=workers,
        checkpoint_time_interval_minutes=1e-12,
    )
    original_save = train.save_checkpoint

    def save(model, config, best_acc, filename, **kw):
        original_save(model, config, best_acc, filename, **kw)
        if config.output_dir != cfg.output_dir:
            return
        kind, epoch = kw["checkpoint_kind"], kw["epoch"]
        if kind == "time" and epoch == 1 and kw["batch_idx"] == 1:
            shutil.copyfile(filename, tmp_path / "partial.pth")
        if kind == "latest" and epoch == 0:
            shutil.copyfile(filename, tmp_path / "complete.pth")
        if kind == "pre_eval" and epoch == 1:
            shutil.copyfile(filename, tmp_path / "pre_eval.pth")

    initialization = 0

    def init_context(config):
        nonlocal initialization
        if world_size == 1:
            return train.DistributedContext(), torch.device("cpu")
        rendezvous = (tmp_path / f"ddp_init_{initialization}").as_uri()
        initialization += 1
        dist.init_process_group(
            "gloo", init_method=rendezvous, rank=rank, world_size=world_size
        )
        return train.DistributedContext(True, rank, rank, world_size), torch.device(
            "cpu"
        )

    with (
        patch.object(train, "get_data_loaders", replay_loaders),
        patch.object(train, "build_model", ReplayModel),
        patch.object(train, "_init_distributed", init_context),
        patch.object(train, "save_checkpoint", save),
        patch.object(train, "evaluate_metrics_distributed", fake_evaluate),
    ):
        train.train_onn_model(cfg)
        expected = torch.load(
            tmp_path / "original/latest_checkpoint.pth", weights_only=False
        )
        best = torch.load(
            tmp_path / "original/fftconv_checkpoint.pth", weights_only=False
        )
        for source in ["partial", "complete", "pre_eval"]:
            resumed = replace(
                cfg,
                output_dir=str(tmp_path / source),
                resume_checkpoint=str(tmp_path / f"{source}.pth"),
            )
            train.train_onn_model(resumed)
            actual = torch.load(
                tmp_path / source / "latest_checkpoint.pth", weights_only=False
            )
            for key in [
                "model_state_dict",
                "optimizer_state_dict",
                "scheduler_state_dict",
                "best_model_state_dict",
                "best_snapshot",
                "global_step",
            ]:
                assert_equal_state(actual[key], expected[key])
            for rank_index in range(world_size):
                a, b = (
                    actual["rank_states"][rank_index],
                    expected["rank_states"][rank_index],
                )
                assert a["running_loss"] == b["running_loss"]
                torch.testing.assert_close(
                    a["rng"]["torch"], b["rng"]["torch"], rtol=0, atol=0
                )
            restored_best = torch.load(
                tmp_path / source / "fftconv_checkpoint.pth", weights_only=False
            )
            assert_equal_state(
                restored_best["model_state_dict"], best["model_state_dict"]
            )
            assert restored_best["epoch"] == best["epoch"] == 0


@pytest.mark.parametrize("workers", [0, 2])
def test_partial_complete_and_pre_eval_replay(tmp_path, workers):
    _exercise_replay(tmp_path, workers)


def _ddp_replay_worker(rank, directory):
    torch.set_num_threads(1)
    _exercise_replay(Path(directory), 0, rank=rank, world_size=2)


def test_distributed_training_replay(tmp_path):
    mp.spawn(_ddp_replay_worker, args=(str(tmp_path),), nprocs=2, join=True)


def test_resume_rejects_unreplayable_checkpoint(tmp_path):
    model = nn.Linear(2, 2)
    path = tmp_path / "old.pth"
    torch.save({"model_state_dict": model.state_dict()}, path)
    with pytest.raises(ValueError, match="training_state_version=2"):
        train.load_training_checkpoint(
            str(path), model, torch.device("cpu"), config=AppConfig()
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"adc_bits": 4},
        {"label_smoothing": 0.2},
        {"optimizer": "sgd"},
        {"num_epochs": 3},
        {"batch_size": 2},
        {"seed": 21},
        {"tia_distortion_strength": 1.0},
        {"jtc_remodulation": {"field_coefficients": [0.5, 0.0]}},
        {"jtc_max_shots": 64},
    ],
    ids=lambda changes: next(iter(changes)),
)
def test_resume_rejects_changed_experiment_before_restoring_state(tmp_path, changes):
    config = AppConfig()
    saved = nn.Linear(3, 2)
    path = tmp_path / "checkpoint.pth"
    save_checkpoint(saved, config, -1.0, str(path), data_seed=17)
    restored = nn.Linear(3, 2)
    initial = {k: v.clone() for k, v in restored.state_dict().items()}
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match=next(iter(changes))):
        load_training_checkpoint(
            str(path), restored, torch.device("cpu"), config=replace(config, **changes)
        )
    assert_equal_state(restored.state_dict(), initial)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


@pytest.mark.parametrize(
    "artifact", ["fftconv_checkpoint.pth", "latest_checkpoint.pth"]
)
def test_pretrained_weights_start_fresh_training_from_either_artifact(
    tmp_path, artifact
):
    config = analog_config(
        seed=17,
        num_epochs=1,
        batch_size=2,
        max_train_batches=2,
        run_pretrain_tests=False,
        run_full_strength_inference=False,
        show_progress=False,
        train_eval_interval=0,
        dataloader_num_workers=0,
        jtc_frontend_snr_db=35.0,
        output_dir=str(tmp_path / "original"),
    )
    with (
        patch.object(train, "get_data_loaders", replay_loaders),
        patch.object(train, "build_model", ReplayModel),
        patch.object(train, "evaluate_metrics_distributed", fake_evaluate),
        patch.object(
            train,
            "_init_distributed",
            lambda _: (DistributedContext(), torch.device("cpu")),
        ),
    ):
        train.train_onn_model(config)
        source = tmp_path / "original" / artifact
        weights = torch.load(source, weights_only=False)["model_state_dict"]
        fresh = replace(
            config,
            num_epochs=2,
            seed=29,
            label_smoothing=0.1,
            learning_rate=0.003,
            output_dir=str(tmp_path / "control"),
        )

        def initialized_model(config):
            model = ReplayModel(config)
            train.load_model_state(model, weights)
            return model

        # Independent control: initialize the model with these weights, then
        # create a fresh optimizer, schedule and RNG/data stream normally.
        with patch.object(train, "build_model", initialized_model):
            train.train_onn_model(fresh)
        train.train_onn_model(
            replace(
                fresh,
                output_dir=str(tmp_path / "finetune"),
                pretrained_weights=str(source),
            )
        )
        expected = torch.load(
            tmp_path / "control/latest_checkpoint.pth", weights_only=False
        )
        actual = torch.load(
            tmp_path / "finetune/latest_checkpoint.pth", weights_only=False
        )
        for name in (
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "best_model_state_dict",
            "best_snapshot",
            "rank_states",
            "data_seed",
            "global_step",
            "epoch",
        ):
            if name == "rank_states":
                # NumPy RNG tuples contain ndarrays rather than tensors.
                for a, b in zip(actual[name], expected[name]):
                    np.testing.assert_equal(a["rng"]["numpy"], b["rng"]["numpy"])
                    for key in ("python", "torch", "cuda"):
                        assert_equal_state(a["rng"][key], b["rng"][key])
            else:
                assert_equal_state(actual[name], expected[name])
        assert actual["epoch"] == 1
        assert actual["global_step"] == 4
