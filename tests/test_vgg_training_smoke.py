import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# Provide a lightweight seaborn stub to avoid optional dependency during tests
if "seaborn" not in sys.modules:
    seaborn_stub = types.SimpleNamespace(set_theme=lambda **kwargs: None)
    sys.modules["seaborn"] = seaborn_stub

# Add repository root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from onn_config import AppConfig
from onn_train import VGG_CONFIGS, build_model


def test_vgg11_training_step_updates_params():
    torch.manual_seed(0)

    # Use a reduced VGG configuration to keep the smoke test lightweight
    original_cfg = VGG_CONFIGS["vgg11"]
    VGG_CONFIGS["vgg11"] = [8, "M", 16, "M", 32]

    try:
        config = AppConfig(
            model="vgg11",
            conv_backend="pytorch",
            input_length=17,
            weight_length=3,
            run_pretrain_tests=False,
            pretrain_tests_only=False,
        )

        model = build_model(config)
        # Replace classifier to match the reduced feature depth
        model.classifier = nn.Linear(32, 10)
        model.train()

        inputs = torch.randn(2, 3, 32, 32)
        labels = torch.randint(0, 10, (2,))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1)

        # One step of training should propagate gradients and update weights
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        first_param = next(model.parameters())
        before_step = first_param.detach().clone()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            outputs_after = model(inputs)
            loss_after = criterion(outputs_after, labels)

        has_updated = torch.any(torch.ne(before_step, first_param)).item()

        assert torch.isfinite(loss).all()
        assert torch.isfinite(loss_after).all()
        assert has_updated, "Expected parameters to update after optimizer step"
        assert loss_after.item() >= 0.0
    finally:
        VGG_CONFIGS["vgg11"] = original_cfg
