# 2D pJTC packaging model (preliminary)

A first-order sizing model for the three 2D packaging architectures of a
100 x 100-channel pJTC stack. It compares TSV/TGV bandwidth, gap crossings,
power delivery, thermal limits, memory placement and throughput against the
1D JTC. Every component number is a placeholder: replace it with PDK, package
and measured data before drawing conclusions beyond relative trends.

```bash
python -m packaging2d.study --output-dir runs/packaging2d    # report + CSVs
python -m packaging2d.study --report packaging2d/RESULTS.md  # refresh results
python -m packaging2d.figures --output-dir packaging2d/figures
pytest -q tests/test_packaging2d.py
```

The model needs only the Python standard library and `onn_shotplan`; the
figures need matplotlib. `model.evaluate(arch, StackParams(...))` returns one
flat record; `model.with_overrides(p, via__kind="TSV", power__heater_mw_per_ring=0.5)`
changes nested parameters. [RESULTS.md](RESULTS.md) holds the generated tables.

## Stack and architectures

All three use the same three layers. The optics are taken as given: a 5 um
pixel pitch on both optical planes and ~500 um between the emitting and
receiving grating couplers. That separation is the middle-layer thickness, so
every vertical via is 500 um long: 50 um diameter on 100 um pitch at 10:1.

| Layer | Optical half | Electronic half |
|---|---|---|
| Top: GF 45SPCLO | GC emitters (5 x 5 um) | MRM + driver + heater |
| Middle: 500 um | glass + 2D metalens | 12LP EIC: DAC, ADC, accumulators, memory PHY |
| Bottom: GF 45SPCLO | GC receivers | PD + TIA |

| Key | Name used here | Your organization | Fourier loop (PD/TIA -> MRM) | Lens passes per shot |
|---|---|---|---|---|
| A | Single-sided package + vertical interconnect | 1: LPDDR5/HBM on the bottom package | crosses the gap | 2, sharing one lens |
| B | Dual-sided packaging | 2: top die also packaged with memory and power | crosses the gap | 2, sharing one lens |
| C | Single-sided electronics + optical loopback | 3: two optical columns, second inverted | stays on the bottom die | 1 (pipelined) |

The mapping of names to organizations is an interpretation; check it before
reusing the numbers. In C, light goes down column 1, is detected and
remodulated on the bottom die, then goes back up column 2 to output PDs on the
top die. The EIC is face-to-face with the top die, so DACs and ADCs reach their
PIC lanes without crossing the gap.

## What the model computes

- **Vias.** Diameter `= gap / AR`, pitch `= 2 x diameter`. TGV capacitance uses
  a two-wire model in glass (er 4.8). TSV capacitance uses the oxide liner
  (2% of the diameter) times 0.7 for depletion. Inductance uses the loop to
  adjacent returns. The -3 dB point of the driver into that R-L-C sets the
  analog lane rate (6-bit settling in half a sample) and the NRZ rate (2 x f3dB).
- **Gap crossings.** A and B must move N^2 Fourier-plane samples across the gap
  every shot. The lanes are direct 1:1, or analog TDM (time-division
  multiplexing) capped at 4 GS/s per lane. When lanes do not fit the gap-via
  budget, they throttle the shot rate. In C only power and serialized memory
  traffic cross. Power vias are sized by a Cu current density of 5e4 A/cm2.
- **Throughput.** The model reuses the repo's contamination rule
  (`onn_shotplan.compute_contamination_profile`). 1D: F = 256, clean
  M = 64, 186 MAC/shot. 2D: kernel and signal are separated along x, which gives
  a clean 25 x 98 box at N = 100 and 19,872 MAC/shot. Speedup is MACs/s against
  the 1D JTC at the same modulation rate. It is averaged over the CIFAR ResNet-18
  stages (32/16/8/4 maps, equal MACs, so a harmonic mean) with packing and
  stitching per `jtc_geometry_constraints.md`.
- **Power.** Per die: heaters, MRM drive, TIA, DAC, ADC (Walden), accumulation,
  memory, gap lanes. The laser is reported separately (off-stack).
- **Thermal.** A three-node network (top, EIC, bottom) with a cold plate on the
  exposed top, an organic package to the board below, a top package in the path
  for B, and conduction through the Si spacer and glass. The model reports the
  smallest footprint that keeps photonic dies at or below 85 C and the EIC at or
  below 105 C, capped at 400 mm2.
- **Memory.** Bits per shot = weights + input tile / C_out reuse + outputs /
  (C_in x signed branches). The result is sized as LPDDR5X (34 GB/s) or
  HBM3 (819 GB/s) devices.

## Findings (defaults, N = 100, 500 um gap)

1. **Via physics.** A 500 um TGV has ~0.1 pF, f3dB ~27 GHz and ~17 GS/s analog
   with a 50 ohm driver (6.9 GS/s at 100 ohm). A TSV of the same size has
   ~1.9 pF of liner capacitance, ~1.6 GHz and ~1.1 GS/s, so TSVs fail every
   10 GHz analog lane.
2. **The A/B Fourier loop sets the via field.** 10k samples per shot need
   1,250 TGV lanes with TDM x8 at 1 GHz: 25 mm2 at AR 10, 6 mm2 at AR 20,
   ~3 W. At 10 GHz the TDM cap forces 1:1 lanes: 200 mm2 at AR 10, 50 mm2 at
   AR 20. Digitizing the loop in the EIC costs 34 W at 1 GHz and 342 W at
   10 GHz. With TSVs, A/B are throttled below 10x at 10 GHz.
3. **C removes the crossing and doubles the shot rate.** The speedup is 73x
   (C) against 37x (A/B). The 10x target needs N >= 40 (C) or N >= 56 (A/B).
   The 1D baseline completes one shot per modulation period; if the 1D chip
   also reuses one lens for both passes, every speedup doubles.
   C pays with 2x rings (heater power), 2x laser and a buried bottom die that
   dissipates 17 W (1 GHz) to 76 W (10 GHz).
4. **Thermal ranks A, C << B.** At 1 GHz the stacks dissipate 26 / 26 / 39 W
   (A / B / C) and need 55 / 144 / 36 mm2. At 10 GHz, B cannot be cooled within
   400 mm2 because its top package sits in the heat path; A needs 150 mm2 with
   AR 20 TGVs and C 177 mm2 at 184 W. Halving TIA and driver energy and
   integrating 8 shots per ADC brings C to 98 W and 95 mm2.
5. **Memory is not the bottleneck when reuse is good.** At 1 GHz A and C
   need one LPDDR5X and B one per package. At 10 GHz A needs 3 LPDDR5X, B 3 on
   top plus 1 below, and C one HBM3 (184 GB/s). In C the memory traffic
   crosses the gap on 3 (1 GHz) to 28 (10 GHz) serialized TGV lanes; how an
   HBM3 attaches to C at 10 GHz is not modelled yet.
6. **Power delivery is not the constraint.** The worst case, C at 10 GHz,
   needs ~250 power vias for 108 W crossing the gap.

## Design points worth carrying forward

| Point | Arch | f_mod | Vias | Speedup | Stack W | Laser W | Footprint |
|---|---|---|---|---|---|---|---|
| DP1 | C | 1 GHz | TGV AR10: power + 3 memory lanes | 73x | 39 | 13 | ~36 mm2 |
| DP2 | C | 10 GHz, low-energy TIA/driver, 8 shots per ADC | TGV AR10 | 73x | 98 | 13 | ~95 mm2 |
| DP3 | A | 1 GHz | TGV AR20: 1,430 lanes, TDM x7 | 37x | 26 | 7 | ~33 mm2 |

A at 10 GHz is also feasible with AR 20 TGVs (125 W, ~150 mm2, 50 mm2 of via
field), or with AR 10 TGVs if the gap-via budget is raised to 250 mm2 (its
field is 201 mm2; at 200 mm2 it throttles to ~15x). B is not carried forward: it keeps the gap-crossing Fourier loop and
adds the worst heat path.

## Limitations

- Placeholder component energies and areas: heaters 1 mW/ring, TIA 0.4 pJ,
  MRM drive 0.25 pJ, ADC FoM 20 fJ/step. These dominate the absolute watts.
- No noise or precision model for analog TDM. No substrate loss, crosstalk or
  depletion bias for TSVs. Via electrical models are lumped.
- The thermal model is 1D per area, with no lateral spreading within a die and
  no ring-to-ring gradients.
- Optics are taken as given (5 um pitch, 500 um separation); lens efficiency
  enters only through the laser loss budget.
