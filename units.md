# Units

| Block | Input | Output | Notes |
| --- | --- | --- | --- |
| DAC | normalized tensor | normalized drive voltage | Converter rails [0,1]; configured DAC bits |
| Driver | drive voltage | V | Input modulator drive |
| Input MRM | V | field amplitude √W, optionally complex | Magnitude squared gives power |
| Optical transform | field amplitude √W | field amplitude √W | Detection computes squared magnitude, with configured loss/FFT normalization |
| PD | optical power W | voltage-like block output | Supplied fit is not a bare-PD current characterization |
| TIA | PD block-output units | normalized detector response | Bias uses PD-output units; physical drive-voltage calibration is an assumption |
| Analog remodulator | V | signed/complex field amplitude √W | Independent continuous transfer; optional voltage rails/noise; no converter |
| Output gain/ADC | dark-subtracted voltage, then normalized full scale | normalized codes [0,1] | One final ADC stage for analog shots; configured ADC bits |
| Digital decoding | normalized ADC codes and gain | √(dark-subtracted voltage) | `sqrt(code)/sqrt(gain)`; accumulated across signed branches |

`bits: null` disables quantization rounding but retains converter saturation.
Remodulation field-polynomial coefficient units are √W/V^k; phase-polynomial
coefficient units are rad/V^k. CSV columns are `voltage_v`, `field_sqrt_w`,
and optional `phase_rad`.

## Physical conventions and source data

For input field `e` in √W and transform-grid length `L`, each optical pass
computes `p = (loss/L) |FFT(e)|²`. In the analog path the first PD/TIA output
continuously drives the remodulator before the second lens. The final readout
subtracts the modeled dark level, applies gain, digitizes and square-root
decodes; signed branches are then accumulated electronically. Nonnegative
inputs need two weight-sign branches; general signed inputs need four.

`jtc_total_field` is a discrete transform-grid length; `jtc_separation` is
the unused gap between apertures. For lengths M and K and gap g, center
separation is `g + (M+K)/2`. Circular correlation includes wraparound, self
terms and the mirrored lobe. Overlapping fields interfere before detection.
Input positions, intermediate samples and ADC lanes are distinct. A proposed
64-position input with 32 mirrored channels needs an explicit sampling contract.
The planner's zero-padded linear-reference FFT does not establish that contract.

The supplied Yen manuscript identifies the component curves as PDK/device/
circuit simulations. Individual CSVs do not identify PDK revision, corner,
temperature or wavelength. Files `mrm_amp_w*_sim_data.csv` store **power in W**;
the loader square-roots samples before fitting field amplitude. They replace
byte-identical `mrm_pwr_w*` files. Phase samples are radians. The PD curve is a
voltage-like block response; its output and TIA bias share units.

Transfer strength uses `Hα = (1−α) Hideal + α Hfit`. The voltage/amplitude
ideal is endpoint-affine; phase ideal is zero. α is a sensitivity parameter,
not calibrated process variation. Automatic fitting accepts the first degree
with R² > 0.9995, otherwise minimum AIC over the full scan; explicit configured
degrees override this. Retain CSV hashes, degrees, clipping, laser power, bias,
gain and ideal-reference choices with results. The source basis is listed in
[README.md](README.md#paper-basis-and-model-limits).

## Noise terms

- `jtc_frontend_snr_db`: output ADC-input-referred RMS noise as a fraction of
  full scale, `10^(-SNR/20)`, applied after gain.
- `jtc_remodulation.voltage_noise_rms_v`: interstage drive noise in V RMS,
  before analog rails and the continuous field/phase transfer.
- `laser_rin_db`: per-shot global laser **RMS fractional intensity** noise in
  dB, `20*log10(rms_fraction)`. Applied in the intensity domain before LER
  splitter variation; field amplitude receives the square root of the scale.
- `pd_noise_w`: additive PD input-referred noise in W RMS, independently
  sampled per input sample at each detector evaluation.

The last two terms describe the emulation backend. The analog shot contract
currently rejects laser RIN, LER variation and PD input noise; use the explicit
supported output and remodulator noise terms for analog studies.
