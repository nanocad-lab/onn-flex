# 2D pJTC packaging model (preliminary)

A first-order sizing model for the three 2D packaging architectures of a
100 x 100-channel pJTC stack. It compares optics, TSV/TGV bandwidth, gap
crossings, power delivery, thermal limits, memory placement and throughput
against the 1D JTC. Every component number is a placeholder: replace it with
PDK, package and measured data before drawing conclusions beyond relative
trends.

```bash
python -m packaging2d.study --output-dir runs/packaging2d    # report + CSVs
python -m packaging2d.study --report packaging2d/RESULTS.md  # refresh results
pytest -q tests/test_packaging2d.py
```

The model needs only the Python standard library and `onn_shotplan`.
`model.evaluate(arch, StackParams(...))` returns one flat record;
`model.with_overrides(p, via__kind="TSV", power__heater_mw_per_ring=0.5)`
changes nested parameters. [RESULTS.md](RESULTS.md) holds the generated tables.

## Stack and architectures

All three use the same three layers. The middle layer's thickness is the
optical gap that every vertical via must span.

| Layer | Optical half | Electronic half |
|---|---|---|
| Top: GF 45SPCLO | GC emitters (5 x 5 um) | MRM + driver + heater |
| Middle: gap | glass + 2D metalens | 12LP EIC: DAC, ADC, accumulators, memory PHY |
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

- **Optics.** DFT-matched Fourier sampling sets `p_in * p_F = (lambda/n) f / N`,
  so `f_eff = n N p_in p_F / lambda`. The gap is `f_eff / G`, where G is the
  compression a folded, telephoto or spaceplate design must supply.
  Pixel options: ideal 5 um (GC-limited, no routing room), in-pixel 10 um and
  15 um (ring/PD inside the pixel), and fan-out (the half-PIC/half-GC layout:
  rows of GCs separated by `N/2` routed waveguides).
- **Vias.** Diameter `= gap / AR`, pitch `= 2 x diameter`. TGV capacitance uses
  a two-wire model in glass (er 4.8). TSV capacitance uses the oxide liner
  (2% of the diameter) times 0.7 for depletion. Inductance uses the loop to
  adjacent returns. The -3 dB point of a 50 ohm driver into that R-L-C sets the
  analog lane rate (6-bit settling in half a sample) and the NRZ rate (2 x f3dB).
- **Gap crossings.** A and B must move N^2 Fourier-plane samples across the gap
  every shot. The lanes are direct 1:1, or analog TDM (time-division
  multiplexing) capped at 4 GS/s per lane. When lanes do not fit the 200 mm2
  via budget, they throttle the shot rate. In C only power and serialized
  memory traffic cross. Power vias are sized by a Cu current density of
  5e4 A/cm2.
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

## Findings (defaults, N = 100)

1. **Optics, not packaging, binds first.** A 500 um gap holds at most 18 x 18
   pixels even at a 5 um pitch. 100 x 100 needs `f_eff` of 2.8 mm (5 um pitch),
   11 mm (10 um in-pixel), 25 mm (15 um in-pixel) or 81 mm (fan-out). The
   half-PIC/half-GC fan-out layout is infeasible at any G <= 8, so the GC
   arrays need in-pixel modulators and detectors at <= 10 um pitch, plus
   G ~ 4 for a ~2.8 mm gap.
2. **Via physics.** At 500 um, a TGV has ~0.1 pF, f3dB ~27 GHz and ~17 GS/s
   analog. A TSV has ~1.9 pF (liner), ~1.6 GHz and ~1 GS/s, so TSVs fail every
   10 GHz analog lane. At gaps of 2 mm or more, TSVs fail at 1 GHz too. Beyond
   ~1.4 mm, a 10 GHz lane is no longer lumped and needs coaxial TGV design.
3. **The A/B Fourier loop is the wall.** 10k samples per shot cost 25 mm2 of
   500 um TGV field at 1 GHz (TDM x8, ~3 W). At 10 GHz the TDM cap forces 1:1
   lanes: 200 mm2 at 500 um and 800 mm2 at 1 mm. Digitizing the loop in the EIC
   costs 34 W at 1 GHz and 342 W at 10 GHz. A/B therefore need gaps of
   <= ~0.5-0.7 mm, and the optics then needs G >= 4-20.
4. **C removes the crossing and doubles the shot rate.** The speedup is 73x
   (C) against 37x (A/B) at N = 100. The 10x target needs N >= 40 (C) or
   N >= 56 (A/B). C pays with 2x rings (heater power), 2x laser and a buried
   bottom die that dissipates 17 W (1 GHz) to 76 W (10 GHz).
5. **Thermal ranks A < C << B.** At 1 GHz the stacks dissipate 26 / 26 / 39 W
   (A / B / C) and need 55 / 144 / 36 mm2. At 10 GHz, B cannot be cooled within
   400 mm2 because its top package sits in the heat path. C at 10 GHz is 184 W
   (177 mm2). Halving the TIA and driver energy and integrating 8 shots before
   each ADC cuts this to ~98 W and 95 mm2.
6. **Memory is not the bottleneck when reuse is good.** 1 GHz needs one
   LPDDR5X. 10 GHz needs 2-3 LPDDR5X (A/B) or one HBM3 (C). In C the memory
   traffic crosses the gap on ~30-100 serialized TGV lanes; flipping C to put
   the package on the EIC side removes that.

## Design points worth carrying forward

| Point | Arch | f_mod | Optics | Gap / vias | Speedup | Stack W | Footprint |
|---|---|---|---|---|---|---|---|
| DP1 | C | 1 GHz | in-pixel 10 um, G = 4 | 2.8 mm glass, TGV AR20 | 73x | 39 (+53 laser) | ~45 mm2 |
| DP2 | C | 10 GHz | same + low-energy TIA/driver, analog accumulation x8 | same | 73x | 98 (+53 laser) | ~140 mm2 |
| DP3 | A | 1 GHz | 5 um pitch, G ~ 5.5 | 0.5 mm, TGV AR10, TDM x8 | 37x | 26 (+7 laser) | ~55 mm2 |

B is not carried forward: it keeps the gap-crossing Fourier loop and adds the
worst heat path. The in-pixel options pay for their low Fourier-plane fill
factor (25% at 10 um) in laser power, which is why DP1 needs a 53 W laser
against 7 W for DP3.

## Limitations

- Placeholder component energies and areas: heaters 1 mW/ring, TIA 0.4 pJ,
  MRM drive 0.25 pJ, ADC FoM 20 fJ/step. These dominate the absolute watts.
- No noise or precision model for analog TDM. No substrate loss, crosstalk or
  depletion bias for TSVs. Via electrical models are lumped (flagged beyond
  0.4 of the 10 GHz quarter-wave).
- The thermal model is 1D per area, with no lateral spreading within a die and
  no ring-to-ring gradients.
- The optics model assumes ideal compression G. Lens efficiency and aberrations
  enter only through the laser loss budget.
