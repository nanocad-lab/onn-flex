"""Replayable data sampling and process RNG state for training checkpoints."""

import random
from contextlib import contextmanager

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


def validate_resume_config(saved, config):
    """Keep the experiment fixed; only operational settings may change.

    Solver/chunk choices remain fixed too: roundoff and noise ordering can
    change even when two execution strategies model the same physics.
    """
    operational = {
        "config_file",
        "output_dir",
        "pretrained_weights",
        "resume_checkpoint",
        "eval_only",
        "checkpoint_interval",
        "checkpoint_time_interval_minutes",
        "run_pretrain_tests",
        "pretrain_tests_only",
        "run_full_strength_inference",
        "dataloader_num_workers",
        "dataloader_pin_memory",
        "dataloader_persistent_workers",
        "dataloader_prefetch_factor",
        "memory_report_interval",
        "show_progress",
    }
    current = vars(config)
    changed = sorted(
        name
        for name in (saved.keys() | current.keys()) - operational
        if name not in saved or name not in current or saved[name] != current[name]
    )
    if changed:
        raise ValueError(
            f"Exact resume requires unchanged experiment settings: {', '.join(changed)}. "
            "Use pretrained_weights to start a new experiment from saved model weights."
        )


@contextmanager
def sample_rng(seed):
    """Isolate augmentation RNG from training and from worker scheduling."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.default_generator.manual_seed(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


class EpochDataset(Dataset):
    """Give each sample a stable augmentation for its epoch."""

    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.seed = int(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        epoch, index = key
        seed = (self.seed + epoch * len(self) + index) % (2**63 - 1)
        with sample_rng(seed):
            return self.dataset[index]


class EpochSampler(DistributedSampler):
    """An epoch permutation with an explicit consumed-sample cursor.

    The cursor is set by the training loop, never by prefetched iteration.
    Length remains the full epoch length for progress and batch limits.
    """

    start_index = 0

    def __iter__(self):
        indices = list(super().__iter__())
        return iter((self.epoch, index) for index in indices[self.start_index :])

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self.start_index = 0


def capture_rng(device):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        if device.type != "cuda":
            raise ValueError(
                "A CUDA training checkpoint requires CUDA for exact resume"
            )
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)
    elif device.type == "cuda":
        raise ValueError("A CPU training checkpoint requires CPU for exact resume")


def capture_rank_states(device, running_loss=0.0, train_samples=0):
    state = {
        "rng": capture_rng(device),
        "running_loss": float(running_loss),
        "train_samples": int(train_samples),
    }
    if not dist.is_initialized():
        return [state]
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, state)
    return states
