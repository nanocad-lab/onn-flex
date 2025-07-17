# Units
I/O units for each block. Thus we can ensure correct I/O amplitude handling

| Block | Input | Ouput | Notes |
| ----- | ----- | ----- | --- |
|  DAC  |  fp32 |  fp32 (V) | quant to/from int4 with fixed fp32 range [0, 1] |
| Driver | V | V | output range adjustable, match MRM directly
|  MRM  |  V | W (complex) | |  
| Lens  | W (complex) | W (real) | amplitude only output |
| PD/TIA | W (real) | V | ?? |
| ADC | V | int4 | fixed range |
