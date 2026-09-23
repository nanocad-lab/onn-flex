"""Analog pJTC shot physics, independent of convolution and training state.

FFT and aperture-pixel DFT implement the same topology. The layer adapter
supplies explicit geometry and gain; this module never updates calibration.
"""

import math
from dataclasses import replace

import torch
from torch import nn

from onn_component import JTC, raise_dynamo_recompile_limit
from onn_math import complex_abs_squared, sqrt_nonnegative_with_finite_grad
from onn_quantization import converter_quantize_ste
from onn_remodulation import AnalogRemodulator, RemodulationSpec
from onn_shotplan import ApertureGeometry


class AnalogJTCShot(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.remodulator = AnalogRemodulator(
            RemodulationSpec(**config.jtc_remodulation)
        )
        self._jtc_cache = nn.ModuleDict()
        self._build_solvers()

    def _build_solvers(self):
        for name in (
            "pixel_dft",
            "real_fft",
            "complex_fft",
            "rowwise",
            "detect_rowwise",
        ):
            function = getattr(self, name)
            if self.config.compile_jtc:
                raise_dynamo_recompile_limit()
                function = torch.compile(function, dynamic=True)
            setattr(self, "_compiled_" + name, function)

    def __getstate__(self):
        state = dict(self.__dict__)
        for name in (
            "pixel_dft",
            "real_fft",
            "complex_fft",
            "rowwise",
            "detect_rowwise",
        ):
            state.pop("_compiled_" + name, None)
        state.pop("_fourier_lag_cache", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._build_solvers()

    def components(self, geometry, ref):
        key = "_".join(
            map(
                str,
                (
                    geometry.input_length,
                    geometry.kernel_length,
                    geometry.total_field,
                    geometry.separation,
                    geometry.output_length,
                ),
            )
        )
        if key not in self._jtc_cache:
            cfg = replace(
                self.config,
                input_length=geometry.input_length,
                kernel_length=geometry.kernel_length,
                output_length=geometry.output_length,
                jtc_total_field=geometry.total_field,
                jtc_separation=geometry.separation,
            )
            self._jtc_cache[key] = JTC(cfg)
        jtc = self._jtc_cache[key]
        dtype = torch.float64 if ref.dtype == torch.float64 else torch.float32
        if (
            jtc.input_field_coeffs.device != ref.device
            or jtc.input_field_coeffs.dtype != dtype
        ):
            jtc.to(device=ref.device, dtype=dtype)
        self.constants(jtc)
        return jtc

    def _prepare(self, signal, kernel, geometry, gain):
        self.validate()
        if signal.ndim != 2 or kernel.ndim != 2 or signal.shape[0] != kernel.shape[0]:
            raise ValueError("Expected paired [shots, aperture_length] tensors")
        if geometry is None:
            geometry = ApertureGeometry(
                self.config.input_length,
                self.config.kernel_length,
                self.config.jtc_total_field,
                self.config.jtc_separation,
                self.config.output_length,
            )
        if (
            signal.shape[-1] != geometry.input_length
            or kernel.shape[-1] != geometry.kernel_length
        ):
            raise ValueError("Shot aperture lengths must match the geometry")
        if gain is None and self.config.jtc_output_gain_mode == "fixed":
            gain = self.config.jtc_output_gain
        elif gain is None and self.config.jtc_output_gain_mode != "per_shot":
            raise ValueError(
                "Supply the calibrated gain explicitly; AnalogJTCShot has no training controller"
            )
        if gain is not None:
            gain = torch.as_tensor(
                gain,
                device=signal.device,
                dtype=torch.float64 if signal.dtype == torch.float64 else torch.float32,
            )
            if not torch.isfinite(gain).all() or not (gain > 0).all():
                raise ValueError("Readout gain must be finite and positive")
        return self.components(geometry, signal), gain

    def forward(self, signal, kernel, *, geometry=None, gain=None):
        jtc, gain = self._prepare(signal, kernel, geometry, gain)
        output, _ = self.evaluate(signal, kernel, jtc, gain, False)
        return output

    def inspect(self, signal, kernel, *, geometry=None, gain=None):
        """Full complex-FFT reference trace, with one final ADC realization.

        Includes voltage rails, remodulation field/phase and normalized ADC
        codes. This is a diagnostic reference, not a second production pass;
        reset RNG when comparing noisy runs, and allow solver roundoff.
        """
        jtc, gain = self._prepare(signal, kernel, geometry, gain)
        return self.complex_fft(signal, kernel, jtc, gain, False, return_trace=True)

    def validate(self) -> None:
        """The analytic backend is exact only for an all-affine JTC.

        Physics assumptions (agreed modeling contract): analog Fourier plane
        (PD/TIA present but no ADC/DAC/quantization there and no modulator
        clamping), ideal-linear transfer curves everywhere, no lens/phase
        distortion, no stochastic noise. The output ADC range is gain-scaled
        to the useful correlation lags (jtc_output_gain_mode).
        """
        if getattr(self, "_analytic_validated", False):
            return
        cfg = self.config
        problems = []
        # driver/mrm-amplitude distortion acts on the INPUT fields only and
        # is fully supported by the closed forms. The Fourier-plane chain
        # (PD/TIA), lens, and MRM phase must stay ideal for the CLOSED
        # forms; the two-FFT full-transfer path (jtc_analog_fourier with
        # jtc_fourier_closed_form=false) computes them exactly and trains.
        full_transfer_path = cfg.conv_backend == "jtc_analog_fourier" and not bool(
            getattr(cfg, "jtc_fourier_closed_form", True)
        )
        if not full_transfer_path:
            if not self.remodulator.spec.affine:
                problems.append(
                    "non-affine remodulation requires jtc_fourier_closed_form=false"
                )
            if int(cfg.jtc_carrier_stopband_bins) != 0:
                problems.append(
                    "jtc_carrier_stopband_bins must be 0 (carrier suppression "
                    "requires conv_backend='jtc_analog_fourier' with "
                    "jtc_fourier_closed_form=false)"
                )
            for name in (
                "pd_distortion_strength",
                "tia_distortion_strength",
                "mrm_phase_distortion_strength",
                "lens_distortion_strength",
            ):
                if float(getattr(cfg, name) or 0.0) != 0.0:
                    problems.append(
                        f"{name} must be 0 (or set jtc_fourier_closed_form="
                        "false to run the full-transfer plane path)"
                    )
        if float(cfg.ler_std_dev or 0.0) != 0.0:
            problems.append("ler_std_dev must be 0")
        if cfg.laser_rin_db is not None:
            problems.append("laser_rin_db must be null")
        if float(cfg.pd_noise_w or 0.0) != 0.0:
            problems.append("pd_noise_w must be 0")
        if cfg.fourier_plane_bits is not None:
            problems.append("fourier_plane_bits must be null")
        if cfg.pd_input_clamp_min_w is not None or cfg.pd_input_clamp_max_w is not None:
            problems.append("pd_input_clamp_{min,max}_w must be null")
        if cfg.scale_output != "none":
            problems.append(
                "scale_output must be 'none' (jtc_output_gain_mode applies)"
            )
        if problems:
            raise ValueError(
                f"conv_backend={cfg.conv_backend!r} violates its modeling contract: "
                + "; ".join(problems)
            )
        self._analytic_validated = True

    @staticmethod
    def detector_coefficients(jtc: JTC) -> tuple[float, float, float, float]:
        """Cache immutable fit coefficients independently of pipeline constants."""
        cached = getattr(jtc, "_analytic_detector_coeffs", None)
        if cached is None:
            a_pd, b_pd = (float(v) for v in jtc.pd.ideal_coeffs)
            a_t, b_t = (float(v) for v in jtc.tia.ideal_coeffs)
            cached = jtc._analytic_detector_coeffs = (a_pd, b_pd, a_t, b_t)
        return cached

    def constants(self, jtc: JTC) -> tuple[float, float, float, float, float]:
        """Compose the affine pipeline into (a_in, b_in, P, C1) plus gain.

        Extracted output before the ADC is v = P * corr^2 + C1, where corr is
        the cross-correlation of the input fields f = a_in*q(x) + b_in.
        """
        cached = getattr(jtc, "_analytic_consts", None)
        if cached is not None:
            return cached
        # The re-modulation stage is analog-affine under the case-2 contract
        # even when the INPUT modulation runs a nonlinear driver/MRM curve,
        # so the pipeline constants always use the ideal affine composition
        # mrm_ideal(driver_ideal(x)).
        a_d, b_d = (float(v) for v in jtc.driver.ideal_coeffs)
        a_m, b_m = (float(v) for v in jtc.mrm.ideal_field_coeffs)
        a_in, b_in = a_m * a_d, a_m * b_d + b_m
        gain = float(jtc.config.laser_power_gain or 1.0) ** 0.5
        a_in, b_in = a_in * gain, b_in * gain
        a_pd, b_pd, a_t, b_t = self.detector_coefficients(jtc)
        n = float(jtc.jtc_total_field)
        k1 = a_t * a_pd * jtc.loss / n
        a_remod, _ = (
            self.remodulator.affine_coefficients((a_in, b_in))
            if self.remodulator.spec.affine
            else (a_in, b_in)
        )
        p = k1 * (a_remod * k1 * n) ** 2
        # TIA input bias enters the (offset-nulled) constant term only.
        c1 = (
            a_t * (b_pd + float(getattr(jtc.config, "tia_input_bias", 0.0) or 0.0))
            + b_t
        )
        jtc._analytic_consts = (a_in, b_in, p, c1, n)
        return jtc._analytic_consts

    def input_field(self, x: torch.Tensor, jtc: JTC) -> torch.Tensor:
        """DAC + input modulation transfer (driver∘MRM amplitude).

        Supports the full blended polynomial at any distortion strength —
        the closed forms only need the input FIELDS, however they were
        produced; only the Fourier-plane chain and re-modulation must stay
        affine (enforced by validate).
        """
        q = converter_quantize_ste(
            x, self.config.dac_bits, self.config.converter_clamp_grad
        )
        coeffs = jtc.input_field_coeffs
        if coeffs.numel() == 2:
            field = coeffs[0] * q + coeffs[1]
        else:
            field = jtc._eval_poly(coeffs, q)
        gain = float(jtc.config.laser_power_gain or 1.0) ** 0.5
        return (gain * field).clamp_min(0.0)

    def correlation_readout(
        self,
        corr: torch.Tensor,
        p: float,
        c1: float,
        gain: torch.Tensor | None,
        need_vmax: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Detector chain on extracted lags: offset-null, gain, ADC, sqrt.

        The detector's known dark level (c1) is nulled before conversion
        (correlated double sampling) and the gain scales the ADC input so the
        useful outputs fill the code range; the gain is divided back out
        after the converter so both signed branches return values in common
        physical units. Pure function (safe under checkpoint recompute);
        also returns the observed pre-gain max when a calibrating gain mode
        needs it (a full reduction otherwise skipped — it costs ~14%).
        """
        v = p * (corr * corr)  # dark level c1 offset-nulled before the ADC
        return self.readout(v, gain, need_vmax)

    def readout(
        self,
        v: torch.Tensor,
        gain: torch.Tensor | None,
        need_vmax: bool,
        *,
        return_trace=False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert dark-subtracted voltage with the shared case-2 ADC model.

        The caller supplies a forward's immutable gain and observation flag.
        Return the pre-gain, pre-noise maximum without updating calibration;
        the layer aggregates all shot maxima after checkpointed work returns.
        Noise RMS is a configured fraction of ADC full scale (after gain).
        """
        v_max = v.detach().amax() if need_vmax else v.new_zeros(())
        if gain is None:  # per-shot AGC over this shot's useful outputs
            g = 1.0 / v.detach().amax(dim=-1, keepdim=True).clamp_min(1e-12)
        else:
            g = gain
        y_adc = g * v
        snr_db = getattr(self.config, "jtc_frontend_snr_db", None)
        if snr_db is not None:
            # ADC-input-referred noise, independent of the gain setting.
            y_adc = y_adc + torch.randn_like(y_adc) * (10.0 ** (-float(snr_db) / 20.0))
        if (
            self.config.jtc_fuse_pointwise
            and y_adc.is_cuda
            and y_adc.dtype == torch.float32
            and g.dtype == y_adc.dtype
            and (self.config.adc_bits is None or self.config.adc_bits <= 63)
            and g.numel() == 1
            and not g.requires_grad
            and not return_trace
        ):
            from onn_fused import adc_readout

            bits = self.config.adc_bits
            output = adc_readout(
                y_adc,
                g,
                0 if bits is None else 2**bits - 1,
                self.config.converter_clamp_grad == "mad",
            )
            return output, v_max
        y = converter_quantize_ste(
            y_adc, self.config.adc_bits, self.config.converter_clamp_grad
        )
        output = sqrt_nonnegative_with_finite_grad(y) / torch.sqrt(g)
        if return_trace:
            return {
                "output": output,
                "pre_gain_voltage_v": v,
                "gain": g,
                "adc_input_normalized": y_adc,
                "adc_codes_normalized": y,
                "adc_clipped": (y_adc < 0) | (y_adc > 1),
            }
        return output, v_max

    def affine_fft(
        self,
        signal: torch.Tensor,
        kernel: torch.Tensor,
        jtc: JTC,
        gain: torch.Tensor | None,
        need_vmax: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Case-2 contract with real plane FFTs (jtc_analog_fourier).

        Identical physics to the analytic backend's validated FFT reference —
        analog Fourier plane (affine PD/TIA, no quantization or clamping),
        offset-nulled gain-scaled output ADC — but computed on the actual
        configured plane, so finite-lens wraparound, autocorrelation
        contamination, and mirror-lobe overlap appear exactly. This is the
        geometry-sensitive trainable path for lens/separation studies.
        """
        with torch.autocast(device_type=signal.device.type, enabled=False):
            a_in, b_in, _, c1, n = self.constants(jtc)
            a_pd, b_pd, a_t, b_t = self.detector_coefficients(jtc)
            wide = torch.float64 if signal.dtype == torch.float64 else torch.float32
            f_s = self.input_field(signal.to(wide), jtc)
            f_k = self.input_field(kernel.to(wide), jtc)
            kernel_length = kernel.shape[-1]
            sep = jtc.jtc_separation
            plane = f_s.new_zeros(signal.shape[0], jtc.jtc_total_field)
            plane[:, :kernel_length] = f_k
            start = kernel_length + sep
            plane[:, start : start + signal.shape[-1]] = f_s

            jps = complex_abs_squared(torch.fft.fft(plane)) * (jtc.loss / n)
            tia_bias = float(self.config.tia_input_bias)
            if self._fuse_affine_plane(jps, jtc):
                field2 = self._affine_plane(jps, jtc, (a_in, b_in))
            else:
                r1 = a_t * (a_pd * jps + b_pd + tia_bias) + b_t
                field2 = self.remodulator(r1, (a_in, b_in))
            power2 = complex_abs_squared(torch.fft.fft(field2)) * (jtc.loss / n)

            idx = jtc.compute_correlation_indices(signal.device)
            # In the affine detector, subtracting c1 cancels both dark and
            # TIA bias exactly. Cancel symbolically so a large DC bias does
            # not erase small fringes through floating-point subtraction.
            v = (a_t * a_pd) * power2.index_select(-1, idx)
            return self.readout(v, gain, need_vmax)

    def needs_full_transfers(self) -> bool:
        """True when the FFT path must run the real component transfers."""
        cfg = self.config
        return (
            not self.remodulator.spec.affine
            or any(
                float(getattr(cfg, name) or 0.0) != 0.0
                for name in (
                    "pd_distortion_strength",
                    "tia_distortion_strength",
                    "mrm_phase_distortion_strength",
                    "lens_distortion_strength",
                )
            )
            or int(getattr(cfg, "jtc_carrier_stopband_bins", 0) or 0) > 0
        )

    def supports_rfft(self) -> bool:
        """Real plane with an identity lens: rfft/lag-GEMM paths are exact."""
        cfg = self.config
        return (
            self.remodulator.spec.real_symmetric
            and float(cfg.mrm_phase_distortion_strength or 0.0) == 0.0
            and float(cfg.lens_distortion_strength or 0.0) == 0.0
        )

    def uses_plane_fft(self) -> bool:
        """True when full transfers run the FFT plane path (not lag-GEMM)."""
        return self.needs_full_transfers() and not bool(
            getattr(self.config, "jtc_fourier_lag_gemm", True)
        )

    def dft_maps(self, row_width: int, jtc: JTC, device: torch.device):
        """Aperture-pixel DFT maps plus extraction-bin DFT maps.

        Lens phase factors are included in both transforms. The first map
        computes the spectrum from nonzero aperture pixels; the second
        maps remodulated Fourier-plane fields to selected detector bins.
        """
        # For real apertures and an identity lens, the JPS and noiseless real
        # remodulation are even. Evaluate each conjugate pair only once, then
        # fold the second transform's cosine weights. Range regularization
        # currently averages over physical pixels, so retain its full plane.
        symmetric = self.supports_rfft() and not float(
            self.config.pd_range_regularization_weight
        )
        cache = getattr(self, "_fourier_lag_cache", None)
        if cache is None:
            cache = self._fourier_lag_cache = {}
        key = (
            int(row_width),
            jtc.kernel_length,
            jtc.output_length,
            jtc.jtc_total_field,
            jtc.jtc_separation,
            str(device),
            symmetric,
        )
        hit = cache.get(key)
        if hit is not None:
            return hit
        n = int(jtc.jtc_total_field)
        sep = int(jtc.jtc_separation)
        kw = int(jtc.kernel_length)
        w = int(row_width)
        s0 = kw + sep
        ext = jtc.compute_correlation_indices(torch.device("cpu")).tolist()

        # Aperture-pixel DFT: the plane is nonzero on only kw + W pixels, so
        # the exact spectrum is X[u] = sum_j f[j] e^{-i 2pi u pos_j / N} — a
        # (kw+W x N) GEMM of the FIELDS, then J = Xr^2 + Xi^2. Squaring like
        # the FFT route keeps the small fringe bins well-conditioned; the
        # earlier lag-domain form (J = R @ Cos) reconstructed fringes as
        # differences of large autocorrelation terms and lost ~3 digits to
        # cancellation. Strictly functional (no in-place ops).
        #
        # Lens support: F diag(e^{i phi}) F^-1 on the spectrum unwraps to a
        # pointwise phase screen ON THE PLANE, so the deterministic lens
        # phases at the aperture positions (stage 1) and at all N positions
        # (stage 2) fold directly into the DFT angles.
        u_all = torch.arange(n, dtype=torch.float64)
        two_pi_over_n = 2.0 * math.pi / n
        positions = torch.tensor(
            list(range(kw)) + list(range(s0, s0 + w)), dtype=torch.float64
        )
        lens_strength = float(self.config.lens_distortion_strength or 0.0)
        if lens_strength != 0.0 and jtc.lens.base_coefs.numel() > 0:
            coefs = jtc.lens.base_coefs.double().cpu() * lens_strength
            phi = (jtc.lens.basis.double().cpu() * coefs.reshape(-1, 1)).sum(dim=0)
        else:
            phi = torch.zeros(n, dtype=torch.float64)
        # fold sqrt(loss/N) into the pixel matrices: J then carries loss/N
        root_scale = math.sqrt(jtc.loss / n)
        pos_idx = positions.to(torch.long)
        first_bins = u_all[: n // 2 + 1] if symmetric else u_all
        ang1 = (
            phi[pos_idx][:, None]
            - two_pi_over_n * positions[:, None] * first_bins[None, :]
        )
        c_pix = torch.cos(ang1) * root_scale
        s_pix = torch.sin(ang1) * root_scale

        ext_bins = torch.tensor(ext, dtype=torch.float64)
        ang2 = phi[:, None] - two_pi_over_n * u_all[:, None] * ext_bins[None, :]
        cos_e = torch.cos(ang2)
        sin_e = torch.sin(ang2)
        # The chunk mean-subtracts field2 before the stage-2 GEMMs (removes
        # the large common mode that cancels catastrophically); these rows
        # add the mean's exact contribution back (they reduce to N*delta(e=0)
        # and 0 for an identity lens).
        sum_ce = cos_e.sum(dim=0)
        sum_se = sin_e.sum(dim=0)
        stop = int(getattr(self.config, "jtc_carrier_stopband_bins", 0) or 0)
        mask = torch.ones(n, dtype=torch.float32)
        if stop > 0:
            mask[: stop + 1] = 0.0
            mask[-stop:] = 0.0
        if symmetric:
            half = n // 2 + 1
            folded_cos = cos_e[:half].clone()
            paired = (n - 1) // 2
            if paired:
                folded_cos[1 : paired + 1] += cos_e[-paired:].flip(0)
            cos_e = folded_cos
            # An even, real second plane has a real spectrum. DC and the
            # even-length Nyquist bin occur once; all other bins occur twice.
            sin_e = None
            sum_se = None
            mask = mask[:half]
        # All DFT GEMMs run in float64: fringe-bin amplitudes are coherent
        # sums of large terms and fp32 accumulation costs ~2.5e-5 relative
        # there; the GEMMs are a small share of the path cost.
        maps = (
            c_pix.to(device=device),
            s_pix.to(device=device),
            mask.to(device),
            cos_e.to(device=device),
            sin_e.to(device=device) if sin_e is not None else None,
            sum_ce.to(device=device),
            sum_se.to(device=device) if sum_se is not None else None,
        )
        cache[key] = maps
        return maps

    def pixel_dft(
        self,
        signal: torch.Tensor,
        kernel: torch.Tensor,
        jtc: JTC,
        c_pix: torch.Tensor,
        s_pix: torch.Tensor,
        mask: torch.Tensor,
        cos_e: torch.Tensor,
        sin_e: torch.Tensor | None,
        sum_ce: torch.Tensor,
        sum_se: torch.Tensor | None,
        gain: torch.Tensor | None,
        need_vmax: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full-transfer readout without planes or FFTs (real or complex fields).

        Same physics as complex_fft: the exact JPS comes from an
        aperture-pixel DFT (the plane has only kw+W nonzero pixels, so the
        spectrum is two small GEMMs of the fields, squared — the same
        well-conditioned route as the FFT), the measured PD/TIA transfers
        apply pointwise at their configured strengths, the carrier
        stop-band masks the JPS, re-modulation uses the configured continuous voltage-to-field response, and the
        second symmetric plane's detection is evaluated directly at the
        extraction bins (one more cosine GEMM) through the same transfers.
        Strictly functional (no in-place ops). Pure function.
        """
        with torch.autocast(device_type=signal.device.type, enabled=False):
            wide = torch.float64 if signal.dtype == torch.float64 else torch.float32
            complex_mod = float(jtc.mrm.phase_strength) > 0.0
            f_s = jtc.input_distortion(signal.to(wide), as_complex=complex_mod)
            f_k = jtc.input_distortion(kernel.to(wide), as_complex=complex_mod)
            f = torch.cat([f_k, f_s], dim=-1)  # aperture pixels, plane order
            if complex_mod:
                f_r, f_i = f.real.double(), f.imag.double()
                x_re = f_r @ c_pix - f_i @ s_pix
                x_im = f_r @ s_pix + f_i @ c_pix
            else:
                f_r = f.double()
                x_re = f_r @ c_pix
                x_im = f_r @ s_pix
            jps = (x_re * x_re + x_im * x_im).to(wide) * mask
            return self.pixel_readout(
                jps, jtc, cos_e, sin_e, sum_ce, sum_se, gain, need_vmax
            )

    def _fuse_affine_plane(self, power, jtc):
        cfg = self.config
        return (
            cfg.jtc_fuse_pointwise
            and power.is_cuda
            and power.dtype == torch.float32
            and float(jtc.pd.strength) <= 0
            and float(jtc.tia.strength) <= 0
            and self.remodulator.spec.affine
            and not cfg.pd_noise_w
            and cfg.pd_input_clamp_min_w is None
            and cfg.pd_input_clamp_max_w is None
            and not cfg.pd_range_regularization_weight
        )

    def _affine_plane(self, power, jtc, baseline):
        from onn_fused import affine_plane

        ap, bp, at, bt = self.detector_coefficients(jtc)
        am, bm = self.remodulator.affine_coefficients(baseline)
        return affine_plane(
            power, [ap, bp, at, bt, float(self.config.tia_input_bias), am, bm]
        )

    def _fuse_detector_chain(self, power, jtc):
        cfg = self.config
        pd = jtc.pd.ideal_coeffs if float(jtc.pd.strength) == 0 else jtc.pd.coeffs
        tia = jtc.tia.ideal_coeffs if float(jtc.tia.strength) == 0 else jtc.tia.coeffs
        return (
            cfg.jtc_fuse_pointwise
            and power.is_cuda
            and power.dtype == torch.float32
            and float(jtc.pd.strength) in (0.0, 1.0)
            and float(jtc.tia.strength) in (0.0, 1.0)
            and not cfg.pd_noise_w
            and cfg.pd_input_clamp_min_w is None
            and cfg.pd_input_clamp_max_w is None
            and not cfg.pd_range_regularization_weight
            and all(
                c.numel() >= 2 and c.dtype == power.dtype and not c.requires_grad
                for c in (pd, tia)
            )
        )

    def _detector_chain(self, power, jtc, remodulation=None):
        from onn_fused import detector_chain

        pd = jtc.pd.ideal_coeffs if float(jtc.pd.strength) == 0 else jtc.pd.coeffs
        tia = jtc.tia.ideal_coeffs if float(jtc.tia.strength) == 0 else jtc.tia.coeffs
        return detector_chain(
            power, pd, tia, float(self.config.tia_input_bias), remodulation
        )

    def pixel_readout(self, jps, jtc, cos_e, sin_e, sum_ce, sum_se, gain, need_vmax):
        """Physical transfers and second DFT, shared by paired and reused spectra."""
        with torch.autocast(device_type=jps.device.type, enabled=False):
            a_in, b_in, _, _, n = self.constants(jtc)
            wide = jps.dtype
            if self._fuse_affine_plane(jps, jtc):
                field2 = self._affine_plane(jps, jtc, (a_in, b_in))
            elif self._fuse_detector_chain(jps, jtc):
                if self.remodulator.spec.affine:
                    field2 = self._detector_chain(
                        jps, jtc, self.remodulator.affine_coefficients((a_in, b_in))
                    )
                else:
                    field2 = self.remodulator(
                        self._detector_chain(jps, jtc), (a_in, b_in)
                    )
            else:
                r1 = jtc._tia_transfer_raw(jtc._pd_transfer_raw(jps))
                field2 = self.remodulator(r1, (a_in, b_in))
            # mean-subtract before the spectrum GEMM: exact for e != 0
            # (sum of cos rows is zero) and corrected via dc_row otherwise;
            # removes the large common mode that cancels catastrophically.
            if sin_e is None:
                # Mean over the full physical plane, including multiplicities.
                # This also preserves DC extraction for odd and even lengths.
                f2_sum = 2 * field2.sum(dim=-1, keepdim=True) - field2[..., :1]
                if int(n) % 2 == 0:
                    f2_sum = f2_sum - field2[..., -1:]
                f2_mean = f2_sum / n
                f2c = (field2 - f2_mean).double()
                g_re = f2c @ cos_e + f2_mean.double() * sum_ce
                power2 = g_re * g_re
            elif torch.is_complex(field2):
                f2_mean = field2.mean(dim=-1, keepdim=True)
                f2c = (field2 - f2_mean).to(torch.complex128)
                mean = f2_mean.to(torch.complex128)
                g_re = (
                    f2c.real @ cos_e
                    - f2c.imag @ sin_e
                    + mean.real * sum_ce
                    - mean.imag * sum_se
                )
                g_im = (
                    f2c.real @ sin_e
                    + f2c.imag @ cos_e
                    + mean.real * sum_se
                    + mean.imag * sum_ce
                )
                power2 = g_re * g_re + g_im * g_im
            else:
                f2_mean = field2.mean(dim=-1, keepdim=True)
                f2c = (field2 - f2_mean).double()
                g_re = f2c @ cos_e + f2_mean.double() * sum_ce
                g_im = f2c @ sin_e + f2_mean.double() * sum_se
                power2 = g_re * g_re + g_im * g_im
            power2 = power2.to(wide) * (jtc.loss / n)
            zero = torch.zeros((), device=jps.device, dtype=wide)
            if self._fuse_detector_chain(power2, jtc):
                v_full = self._detector_chain(power2, jtc)
                dark = self._detector_chain(zero, jtc)
            else:
                v_full = jtc._tia_transfer_raw(jtc._pd_transfer_raw(power2))
                dark = jtc._tia_transfer_raw(jtc._pd_transfer_raw(zero))
            v = v_full - dark
            return self.readout(v, gain, need_vmax)

    def real_fft(
        self,
        signal: torch.Tensor,
        kernel: torch.Tensor,
        jtc: JTC,
        gain: torch.Tensor | None,
        need_vmax: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """FFT plane path exploiting Hermitian symmetry (real plane only).

        Numerically identical physics to complex_fft: a real
        plane's power spectrum is symmetric (|X[N-k]|^2 = |X[k]|^2 exactly),
        so rfft computes half the bins and every pointwise transfer pass
        runs on N/2+1 values; the symmetric field2 is mirrored back by a
        gather before the second rfft, and extraction folds indices to
        min(idx, N-idx). Fallback when lag-GEMM is disabled.
        """
        with torch.autocast(device_type=signal.device.type, enabled=False):
            a_in, b_in, _, _, n = self.constants(jtc)
            wide = torch.float64 if signal.dtype == torch.float64 else torch.float32
            f_s = jtc.input_distortion(signal.to(wide), as_complex=False)
            f_k = jtc.input_distortion(kernel.to(wide), as_complex=False)
            kernel_length = kernel.shape[-1]
            sep = jtc.jtc_separation
            plane = f_s.new_zeros(signal.shape[0], jtc.jtc_total_field)
            plane[:, :kernel_length] = f_k
            start = kernel_length + sep
            plane[:, start : start + signal.shape[-1]] = f_s

            n_int = int(jtc.jtc_total_field)
            half = n_int // 2 + 1
            mirror = torch.minimum(
                torch.arange(n_int, device=signal.device),
                n_int - torch.arange(n_int, device=signal.device),
            )
            spec = torch.fft.rfft(plane, dim=-1)
            jps_h = (spec.real * spec.real + spec.imag * spec.imag) * (jtc.loss / n)
            stop = int(getattr(self.config, "jtc_carrier_stopband_bins", 0) or 0)
            if stop > 0:
                m = torch.ones(half, device=signal.device, dtype=jps_h.dtype)
                m[: stop + 1] = 0.0  # folded: bins 0..K cover +-K
                jps_h = jps_h * m
            r1_h = jtc._tia_transfer_raw(jtc._pd_transfer_raw(jps_h))
            field2 = self.remodulator(r1_h, (a_in, b_in)).index_select(-1, mirror)
            spec2 = torch.fft.rfft(field2, dim=-1)
            power2_h = (spec2.real * spec2.real + spec2.imag * spec2.imag) * (
                jtc.loss / n
            )
            v_h = jtc._tia_transfer_raw(jtc._pd_transfer_raw(power2_h))
            dark = jtc._tia_transfer_raw(
                jtc._pd_transfer_raw(torch.zeros((), device=signal.device, dtype=wide))
            )
            idx = jtc.compute_correlation_indices(signal.device)
            idx_folded = torch.minimum(idx, n_int - idx)
            v = v_h.index_select(-1, idx_folded) - dark
            return self.readout(v, gain, need_vmax)

    def complex_fft(
        self,
        signal: torch.Tensor,
        kernel: torch.Tensor,
        jtc: JTC,
        gain: torch.Tensor | None,
        need_vmax: bool,
        *,
        return_trace=False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Case-2 contract with the REAL component transfers (trainable).

        Same plane build and output stage as affine_fft, but
        the first detection runs the measured PD/TIA curves at their
        configured strengths, the lens phase screen applies inside
        fft_and_power, and MRM phase makes the input fields complex. The
        optional carrier stop-band nulls the JPS pedestal bins before the
        detector so nonlinear curves act within their fitted input domain
        (physically a DC block at the Fourier plane). Re-modulation uses its independent continuous transfer and the output stage keeps the
        offset-null + gain + ADC + sqrt contract, with the dark level
        evaluated numerically through the same transfers. All transfers are
        differentiable, so this path trains (~FFT-path cost).
        """
        with torch.autocast(device_type=signal.device.type, enabled=False):
            a_in, b_in, _, _, n = self.constants(jtc)
            complex_mod = float(jtc.mrm.phase_strength) > 0.0
            wide = torch.float64 if signal.dtype == torch.float64 else torch.float32
            f_s = jtc.input_distortion(signal.to(wide), as_complex=complex_mod)
            f_k = jtc.input_distortion(kernel.to(wide), as_complex=complex_mod)
            kernel_length = kernel.shape[-1]
            sep = jtc.jtc_separation
            plane = torch.zeros(
                signal.shape[0],
                jtc.jtc_total_field,
                dtype=f_s.dtype,
                device=signal.device,
            )
            plane[:, :kernel_length] = f_k
            start = kernel_length + sep
            plane[:, start : start + signal.shape[-1]] = f_s

            # Inline, branch-free plane FFTs: fft_and_power's rfft fast-path
            # gate is data/shape-dependent, which destabilizes the
            # compile+checkpoint recompute graph on some GPU generations
            # (34-vs-32 saved-tensor mismatch on sm_100).
            def _plane_power(field: torch.Tensor) -> torch.Tensor:
                spec = torch.fft.fft(field, dim=-1)
                if not jtc.lens.is_identity():
                    spec = jtc.lens(spec)
                return spec.real * spec.real + spec.imag * spec.imag

            jps = _plane_power(plane) * (jtc.loss / n)
            stop = int(getattr(self.config, "jtc_carrier_stopband_bins", 0) or 0)
            if stop > 0:
                mask = torch.ones(jps.shape[-1], device=jps.device, dtype=jps.dtype)
                mask[: stop + 1] = 0.0
                mask[-stop:] = 0.0
                jps = jps * mask
            # Raw analog PD->TIA transfers only: the case-2 contract has no
            # mid-plane converter (jtc._detector_power_transfer bakes in an
            # ADC with code-range clipping — that is the emulation contract).
            r1 = jtc._tia_transfer_raw(jtc._pd_transfer_raw(jps))
            if return_trace:
                field2, remod_trace = self.remodulator(
                    r1, (a_in, b_in), return_trace=True
                )
            else:
                field2 = self.remodulator(r1, (a_in, b_in))
            power2 = _plane_power(field2) * (jtc.loss / n)
            v_full = jtc._tia_transfer_raw(jtc._pd_transfer_raw(power2))
            dark = jtc._tia_transfer_raw(
                jtc._pd_transfer_raw(torch.zeros((), device=signal.device, dtype=wide))
            )

            idx = jtc.compute_correlation_indices(signal.device)
            if (
                float(self.config.pd_distortion_strength or 0.0) == 0.0
                and float(self.config.tia_distortion_strength or 0.0) == 0.0
            ):
                a_pd = float(jtc.pd.ideal_coeffs[0])
                a_t = float(jtc.tia.ideal_coeffs[0])
                v = (a_t * a_pd) * power2.index_select(-1, idx)
            else:
                v = v_full.index_select(-1, idx) - dark
            if return_trace:
                return {
                    "first_plane_power_w": jps,
                    "first_detector_voltage_v": r1,
                    "remodulation": remod_trace,
                    "second_plane_power_w": power2,
                    "output_dark_voltage_v": dark,
                    "extraction_indices": idx,
                    "readout": self.readout(v, gain, need_vmax, return_trace=True),
                    "solver": "complex_fft_reference",
                }
            return self.readout(v, gain, need_vmax)

    def aperture_spectrum(self, field, jtc, *, kernel):
        """FP64 aperture DFT with placement and lens phase, before shot fanout.

        Leading dimensions enumerate distinct modulated apertures. This is
        local autograd state, never a module cache. Only DFT paths split the
        transform: FP32 FFT rounding can move downstream ADC code boundaries.
        """
        with torch.autocast(device_type=field.device.type, enabled=False):
            c_pix, s_pix, *_ = self.dft_maps(jtc.input_length, jtc, field.device)
            kw = int(jtc.kernel_length)
            c = c_pix[:kw] if kernel else c_pix[kw:]
            s = s_pix[:kw] if kernel else s_pix[kw:]
            if torch.is_complex(field):
                re, im = field.real.double(), field.imag.double()
                x_re = re @ c - im @ s
                x_im = re @ s + im @ c
            else:
                x_re = field.double() @ c
                x_im = field.double() @ s
            return x_re, x_im

    def spectrum_readout(self, re, im, jtc, wide, gain, need_vmax):
        """Detect the SUM of aperture fields, preserving every interference term."""
        with torch.autocast(device_type=re.device.type, enabled=False):
            _, _, mask, *maps = self.dft_maps(jtc.input_length, jtc, re.device)
            jps = (re * re + im * im).to(wide) * mask
            return self.pixel_readout(jps, jtc, *maps, gain, need_vmax)

    def prepare_rowwise(self, signal, kernel, jtc, nonnegative):
        """Prepare compact aperture spectra before their per-shot expansion."""
        wide = torch.float64 if signal.dtype == torch.float64 else torch.float32

        def prepare(values, *, kernel):
            with torch.autocast(device_type=values.device.type, enabled=False):
                field = jtc.input_distortion(
                    values.to(wide), as_complex=float(jtc.mrm.phase_strength) > 0.0
                )
                return self.aperture_spectrum(field, jtc, kernel=kernel)

        # Batch the two deterministic kernel branches into one modulation/DFT
        # invocation. Detection remains separate to preserve shot/noise order.
        kr, ki = prepare(
            torch.stack((kernel.clamp_min(0), (-kernel).clamp_min(0))), kernel=True
        )
        if nonnegative:
            sr, si = prepare(signal, kernel=False)
        else:
            sr, si = prepare(
                torch.stack((signal.clamp_min(0), (-signal).clamp_min(0))), kernel=False
            )
        return sr, si, kr, ki

    def detect_rowwise(self, sr, si, kr, ki, jtc, wide, gain, need_vmax, nonnegative):
        """Expanded detection planes; independently checkpointable after preparation."""
        kp, kn = (kr[0], ki[0]), (kr[1], ki[1])
        if nonnegative:
            sp = sr, si
        else:
            sp, sn = (sr[0], si[0]), (sr[1], si[1])

        def detect(s, k):
            if self.config.jtc_fuse_pointwise and s[0].is_cuda:
                from onn_fused import spectrum_intensity

                _, _, mask, *maps = self.dft_maps(jtc.input_length, jtc, s[0].device)
                jps = spectrum_intensity(*s, *k, mask, wide)
                return self.pixel_readout(jps, jtc, *maps, gain, need_vmax)
            re = s[0][:, :, None] + k[0][None]
            im = s[1][:, :, None] + k[1][None]
            return self.spectrum_readout(
                re.reshape(-1, re.shape[-1]),
                im.reshape(-1, im.shape[-1]),
                jtc,
                wide,
                gain,
                need_vmax,
            )

        pp, vpp = detect(sp, kp)
        if nonnegative:
            pn, vpn = detect(sp, kn)
            return pp - pn, torch.maximum(vpp, vpn)
        nn, vnn = detect(sn, kn)
        pn, vpn = detect(sp, kn)
        np, vnp = detect(sn, kp)
        return (pp + nn) - (pn + np), torch.maximum(
            torch.maximum(vpp, vnn), torch.maximum(vpn, vnp)
        )

    def rowwise(self, signal, kernel, jtc, gain, need_vmax, nonnegative):
        """Prepare unique apertures, then detect every physical signed shot.

        signal: [positions, inputs, width]; kernel: [inputs, outputs, width].
        Spectra are live autograd values, never a cache across forwards.
        """
        spectra = self.prepare_rowwise(signal, kernel, jtc, nonnegative)
        wide = torch.float64 if signal.dtype == torch.float64 else torch.float32
        return self.detect_rowwise(*spectra, jtc, wide, gain, need_vmax, nonnegative)

    def evaluate(self, signal, kernel, jtc, gain, need_vmax):
        """One production analog shot, shared by row, dot, and diagnostic callers."""
        if self.needs_full_transfers():
            if bool(getattr(self.config, "jtc_fourier_lag_gemm", True)):
                # covers lens and MRM-phase too: lens phases fold into
                # the DFT matrices, complex fields use the 4-GEMM form.
                maps = self.dft_maps(signal.shape[-1], jtc, signal.device)
                return self._compiled_pixel_dft(
                    signal, kernel, jtc, *maps, gain, need_vmax
                )
            if self.supports_rfft():
                return self._compiled_real_fft(signal, kernel, jtc, gain, need_vmax)
            return self._compiled_complex_fft(signal, kernel, jtc, gain, need_vmax)
        return self.affine_fft(signal, kernel, jtc, gain, need_vmax)
