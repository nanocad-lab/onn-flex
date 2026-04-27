# Units
I/O units for each block.

| Block | Input | Output | Notes |
| ----- | ----- | ----- | --- |
|  DAC  |  fp32 |  fp32 (V) | quant to/from int4 with fixed fp32 range [0, 1] |
| Driver | V | V | output range adjustable, match MRM directly
|  MRM  |  V | W (complex) | |
| Lens  | W (complex) | W (real) | amplitude only output |
| PD | W (real) | normalized current | input-referred noise is in W RMS |
| TIA | normalized current | V | |
| ADC | V | int4 | fixed range |

## Noise terms

- `laser_rin_db`: per-shot global laser **RMS fractional intensity** noise in dB, interpreted as `laser_rin_db = 20*log10(rms_fraction)`. Applied as a single multiplicative scale on MRM power **before** the LER splitter variation, so the splitter tree distributes the noisy power unevenly across channels.
- `pd_noise_w`: additive PD input-referred noise in **Watts RMS**, sampled independently per input sample each time the PD is evaluated.
