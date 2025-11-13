"""
Simple smoke test to validate config changes without requiring PyTorch.
Tests configuration validation and parameter handling.
"""

from onn_config import AppConfig


def test_config_backend_validation():
    """Test that backend validation works correctly."""
    print("Testing config backend validation...")

    # Test valid backends
    for backend in ["pytorch", "fourier", "jtc_emulation"]:
        config = AppConfig(conv_backend=backend)
        assert config.conv_backend == backend
        print(f"  ✓ Valid backend '{backend}' accepted")

    # Test invalid backend
    try:
        config = AppConfig(conv_backend="invalid_backend")
        print("  ✗ ERROR: Invalid backend should have raised ValueError")
        return False
    except ValueError as e:
        assert "Invalid conv_backend" in str(e)
        print(f"  ✓ Invalid backend correctly rejected: {e}")

    # Test None backend (should be allowed as default)
    config = AppConfig(conv_backend=None)
    assert config.conv_backend is None
    print("  ✓ None backend allowed (will fallback to jtc_emulation)")

    return True


def test_config_defaults():
    """Test that default configuration is correct."""
    print("\nTesting config defaults...")

    config = AppConfig()

    # Check default backend
    assert config.conv_backend == "jtc_emulation"
    print(f"  ✓ Default backend: {config.conv_backend}")

    # Check that old flags don't exist
    assert not hasattr(config, "use_pytorch_conv")
    assert not hasattr(config, "use_fourier_conv")
    print("  ✓ Old flags (use_pytorch_conv, use_fourier_conv) removed")

    # Check other important defaults
    assert config.jtc_half_size == 8
    assert config.jtc_separation == 0
    assert config.jtc_total_field == 16
    assert config.dac_bits == 4
    assert config.adc_bits == 6
    print("  ✓ Other config defaults intact")

    return True


def test_config_from_yaml_structure():
    """Test that config can be created with all parameters."""
    print("\nTesting config creation with all parameters...")

    config_dict = {
        "conv_backend": "fourier",
        "jtc_half_size": 8,
        "jtc_separation": 8,
        "jtc_total_field": 48,
        "dac_bits": 4,
        "adc_bits": 6,
        "fourier_plane_bits": 6,
        "quantizer": "ste_clipped",
    }

    config = AppConfig(**config_dict)

    assert config.conv_backend == "fourier"
    assert config.jtc_half_size == 8
    assert config.jtc_separation == 8
    assert config.jtc_total_field == 48
    print("  ✓ Config created successfully from dict")

    return True


def test_backend_choices():
    """Test all three backend choices."""
    print("\nTesting all backend choices...")

    backends = ["pytorch", "fourier", "jtc_emulation"]

    for backend in backends:
        config = AppConfig(
            conv_backend=backend,
            jtc_half_size=8,
            jtc_separation=8,
            jtc_total_field=48,
        )
        assert config.conv_backend == backend
        print(f"  ✓ Backend '{backend}' configured successfully")

    return True


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("Running configuration validation smoke tests")
    print("=" * 60)

    tests = [
        test_config_backend_validation,
        test_config_defaults,
        test_config_from_yaml_structure,
        test_backend_choices,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ Test failed with exception: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
