"""Physical shot mapping and accounting, independent of tensor solvers."""

import functools
import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ApertureGeometry:
    input_length: int
    kernel_length: int
    total_field: int
    separation: int
    output_length: int | None = None

    def __post_init__(self):
        if (
            min(self.input_length, self.kernel_length, self.total_field) <= 0
            or self.separation < 0
        ):
            raise ValueError(
                "Aperture lengths/field must be positive and separation nonnegative"
            )
        if self.input_length + self.kernel_length + self.separation > self.total_field:
            raise ValueError(
                "infeasible shot geometry: apertures do not fit the physical field"
            )
        if self.output_length is None:
            object.__setattr__(
                self, "output_length", self.input_length + self.kernel_length - 1
            )
        if self.output_length <= 0:
            raise ValueError("Extraction window must contain at least one sample")

    @property
    def extraction_indices(self):
        start = self.separation + self.kernel_length // 2
        if self.output_length == self.input_length:
            start += 1
        return tuple((start + i) % self.total_field for i in range(self.output_length))

    def contamination(self, offsets):
        """Structural overlap at selected detector bins, independent of signal values.

        This identifies possible interference, not its magnitude or accuracy cost.
        """
        m, k, n, d = (
            self.input_length,
            self.kernel_length,
            self.total_field,
            self.kernel_length + self.separation,
        )
        auto = {i % n for i in range(-(m - 1), m)} | {i % n for i in range(-(k - 1), k)}
        desired = {i % n for i in range(d - (k - 1), d + m)}
        mirror = {i % n for i in range(-(d + m - 1), -(d - (k - 1)) + 1)}
        indices = self.extraction_indices
        return tuple(
            {
                "offset": offset,
                "detector_bin": indices[offset],
                "desired": indices[offset] in desired,
                "autocorrelation": indices[offset] in auto,
                "mirror": indices[offset] in mirror,
                "clean": indices[offset] in desired
                and indices[offset] not in auto
                and indices[offset] not in mirror,
            }
            for offset in offsets
        )


@dataclass(frozen=True)
class ShotPlan:
    mapping: str
    input_shape: tuple[int, int, int, int]
    output_shape: tuple[int, int, int, int]
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    dilation: tuple[int, int]
    padding: tuple[int, int, int, int]
    groups: int
    geometry: ApertureGeometry | None
    branches: tuple[str, ...]
    selected_offsets: tuple[int, ...]
    converted_offsets: tuple[int, ...]
    shots: int
    contributions_per_output: int
    geometry_source: str

    @property
    def adc_samples(self):
        return self.shots * len(self.converted_offsets)

    def to_dict(self):
        record = asdict(self)
        structural = (
            ()
            if self.geometry is None
            else self.geometry.contamination(self.selected_offsets)
        )
        clean = sum(item["clean"] for item in structural)
        record.update(
            {
                "schema_version": 1,
                "adc_stages_per_shot": 1 if self.shots else 0,
                "adc_samples": self.adc_samples,
                "shots_per_input": self.shots // self.input_shape[0],
                "adc_samples_per_input": self.adc_samples // self.input_shape[0],
                "used_adc_samples": self.shots * len(self.selected_offsets),
                "dac_samples_without_reuse": (
                    self.shots
                    * (self.geometry.input_length + self.geometry.kernel_length)
                    if self.geometry
                    else 0
                ),
                "selected_lag_profile": structural,
                "clean_selected_lags": clean,
                "contaminated_selected_lags": len(structural) - clean,
                "contaminated_fraction": (
                    (len(structural) - clean) / len(structural) if structural else 0.0
                ),
                "accuracy_impact": "requires workload evaluation",
                "latency_energy": "requires lane, clock, reuse and component-cost assumptions",
            }
        )
        record["plan_id"] = hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()[:16]
        return record


def resolve_geometry(
    mapping, input_length, kernel_length, total_field, separation, source
):
    if source not in {"config", "auto"}:
        raise ValueError("Geometry source must be 'config' or 'auto'")
    if mapping not in {"row", "dot"}:
        raise ValueError("Shot mapping must be 'row' or 'dot'")
    if source == "auto":
        if mapping == "row":
            total_field, separation = 256, input_length - kernel_length
        else:
            total_field = max(
                total_field, 8 * input_length + 64, 2 * input_length + separation + 1
            )
    return ApertureGeometry(input_length, kernel_length, total_field, separation)


def plan_convolution(
    *,
    input_shape,
    out_channels,
    kernel_size,
    stride,
    dilation,
    padding,
    groups,
    mapping,
    nonnegative,
    total_field,
    separation,
    geometry_source="config",
    readout="full_window",
):
    input_shape = tuple(input_shape)
    kernel_size, stride, dilation, padding = map(
        tuple, (kernel_size, stride, dilation, padding)
    )
    if (
        len(input_shape) != 4
        or any(len(pair) != 2 for pair in (kernel_size, stride, dilation))
        or len(padding) != 4
    ):
        raise ValueError(
            "Expected BCHW, dimension pairs, and top/bottom/left/right padding"
        )
    if (
        min(*input_shape, *kernel_size, *stride, *dilation, out_channels, groups) <= 0
        or min(padding) < 0
    ):
        raise ValueError("Invalid convolution dimensions or padding")
    if readout not in {"full_window", "valid_lags"}:
        raise ValueError("Unknown readout policy")
    b, cin, h, w = map(int, input_shape)
    kh, kw = kernel_size
    sh, sw = stride
    dh, dw = dilation
    top, bottom, left, right = padding
    oh = (h + top + bottom - dh * (kh - 1) - 1) // sh + 1
    ow = (w + left + right - dw * (kw - 1) - 1) // sw + 1
    if (
        min(b, cin, out_channels, oh, ow, groups) <= 0
        or cin % groups
        or out_channels % groups
    ):
        raise ValueError("Invalid convolution workload")
    if mapping not in {"row", "dot"}:
        raise ValueError("Shot mapping must be 'row' or 'dot'")
    branches = (
        ("positive_weight", "negative_weight")
        if nonnegative
        else ("pp", "nn", "pn", "np")
    )
    if kernel_size == (1, 1):
        return ShotPlan(
            "electronic",
            tuple(input_shape),
            (b, out_channels, oh, ow),
            kernel_size,
            stride,
            dilation,
            padding,
            groups,
            None,
            (),
            (),
            (),
            0,
            0,
            "electronic_1x1",
        )
    if mapping == "row":
        if groups != 1 or dilation != (1, 1):
            raise ValueError(
                "Row mapping currently requires groups=1 and dilation=1; explicitly select jtc_shot_mapping='dot' for this workload"
            )
        m, k = w + left + right, kw
        geometry = resolve_geometry(
            mapping, m, k, total_field, separation, geometry_source
        )
        start = (k + 1) // 2 if k > 1 else 0
        offsets = tuple(start + i * sw for i in range(ow))
        shots = b * out_channels * cin * kh * oh * len(branches)
        contributions = cin * kh * len(branches)
        # The fused analytic path converts strided valid outputs directly;
        # its general row implementation converts every valid column.
        if readout == "valid_lags":
            converted = (
                offsets
                if 1 < kw <= 7 and kh <= 7
                else tuple(range(start, start + m - k + 1))
            )
        else:
            converted = tuple(range(geometry.output_length))
    else:
        m = k = (cin // groups) * kh * kw
        geometry = resolve_geometry(
            mapping, m, k, total_field, separation, geometry_source
        )
        offsets = ((m + 1) // 2,)
        converted = (
            offsets if readout == "valid_lags" else tuple(range(geometry.output_length))
        )
        shots = b * out_channels * oh * ow * len(branches)
        contributions = len(branches)
    return ShotPlan(
        mapping,
        tuple(input_shape),
        (b, out_channels, oh, ow),
        kernel_size,
        stride,
        dilation,
        padding,
        groups,
        geometry,
        branches,
        offsets,
        converted,
        shots,
        contributions,
        geometry_source,
    )


@functools.lru_cache(maxsize=65536)
def compute_contamination_profile(
    input_len: int, kernel_len: int, lens_size: int, sep: int
) -> tuple[int, int, int]:
    """Compute contamination profile for JTC configuration.

    Physics: JTC output contains autocorrelation terms plus two mirrored
    cross-correlation lobes. A valid output is clean only if the extracted lag
    is in the desired cross-correlation support and is not overlapped by either
    autocorrelation or the mirrored cross-correlation support.

    Returns:
        total_outputs: Total M+N-1 correlation outputs
        clean_valid_outputs: Number of clean valid convolution outputs (subset of M-N+1)
        effective_stride: Contiguous clean prefix usable for tile stitching
    """
    M, N = input_len, kernel_len

    if M <= 0 or N <= 0 or lens_size <= 0:
        return 0, 0, 0
    if M < N:
        return 0, 0, 0
    if sep < 0:
        return 0, 0, 0
    if M + N + sep > lens_size:
        return 0, 0, 0

    total_outputs = M + N - 1

    geometry = ApertureGeometry(M, N, lens_size, sep)
    valid_start = 0 if N == 1 else (N + 1) // 2
    profile = geometry.contamination(range(valid_start, valid_start + M - N + 1))
    clean_flags = [item["clean"] for item in profile]

    clean_valid_count = sum(clean_flags)
    effective_stride = 0
    for is_clean in clean_flags:
        if not is_clean:
            break
        effective_stride += 1

    return total_outputs, clean_valid_count, effective_stride


@dataclass(frozen=True)
class LinearReadoutTile:
    """A horizontal phase tile with explicit ADC and digital scatter groups."""

    kernel_phase: int
    active_kernel_taps: int
    input_start: int
    input_stride: int
    input_count: int
    output_groups: tuple[tuple[int, ...], ...]
    correlation_groups: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class LinearReadoutPlan:
    """Ideal linear-correlation readout; averaging preserves logical shape."""

    input_shape: tuple[int, int, int, int]
    output_shape: tuple[int, int, int, int]
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int, int, int]
    nonnegative: bool
    readout: str
    signal_slots: int
    kernel_slots: int
    dark_positions: int
    tiles: tuple[LinearReadoutTile, ...]

    @property
    def branch_shots_per_tile(self):
        b, ci, _, _ = self.input_shape
        _, co, oh, _ = self.output_shape
        return b * ci * co * self.kernel_size[0] * oh * (2 if self.nonnegative else 4)

    @property
    def shots(self):
        return self.branch_shots_per_tile * len(self.tiles)

    @property
    def adc_samples(self):
        return self.branch_shots_per_tile * sum(
            len(tile.correlation_groups) for tile in self.tiles
        )

    def to_dict(self):
        result = asdict(self)
        result.update(
            schema_version=1,
            mapping="linear_polyphase_row_readout",
            sampling_assumption="Selected full linear correlation lags available without contamination",
            physical_input_positions=self.signal_slots
            + self.dark_positions
            + self.kernel_slots,
            full_correlations_per_shot=self.signal_slots + self.kernel_slots - 1,
            kernel_loading="Active phase taps first, remaining kernel slots zero",
            detector_pairing="Adjacent lags in the same phase/channel/kernel-row/signed-branch shot",
            digital_scatter=(
                "One mean copied to both logical output positions"
                if self.readout == "pair_mean_hold"
                else "One sample per logical output position"
            ),
            odd_endpoint="Read a final unpaired valid lag individually; requires singleton bypass",
            readout_hardware="Selected individual lags or selectable adjacent means; not fixed pooling of every full-window channel",
            adc_stages_per_shot=1,
            shots=self.shots,
            adc_samples=self.adc_samples,
            dac_samples_without_reuse=self.shots
            * (self.signal_slots + self.kernel_slots),
            max_selected_adc_lanes=max(len(t.correlation_groups) for t in self.tiles),
            excluded="Quantization, noise, power/RMS averaging, transfer distortion, circular aliasing, electronic operations",
        )
        result["plan_id"] = hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()[:16]
        return result


def plan_linear_readout(
    *,
    input_shape,
    out_channels,
    kernel_size,
    stride=(1, 1),
    padding=(0, 0, 0, 0),
    nonnegative=True,
    readout="individual",
    signal_slots=21,
    kernel_slots=21,
    dark_positions=22,
):
    """Preserve conv2d shape while planning individual or pair-mean/hold readout.

    Horizontal stride uses polyphase decomposition in BOTH arms: pack each
    x[r::stride] signal and w[r::stride] kernel. Adjacent logical outputs then
    correspond to adjacent detector lags, even for stride-two convolutions.
    Average inside each shot and copy the decoded mean to both output columns.
    All branches/phases/input channels/kernel rows accumulate digitally.
    """
    input_shape, kernel_size, stride, padding = map(
        tuple, (input_shape, kernel_size, stride, padding)
    )
    if (
        len(input_shape) != 4
        or len(kernel_size) != 2
        or len(stride) != 2
        or len(padding) != 4
    ):
        raise ValueError(
            "Expected BCHW, kernel/stride pairs, and top/bottom/left/right padding"
        )
    if readout not in {"individual", "pair_mean_hold"}:
        raise ValueError("Readout must be individual or pair_mean_hold")
    if (
        min(
            *input_shape,
            out_channels,
            *kernel_size,
            *stride,
            signal_slots,
            kernel_slots,
        )
        <= 0
        or min(*padding, dark_positions) < 0
    ):
        raise ValueError("Invalid linear readout dimensions")
    b, ci, h, w = input_shape
    kh, kw = kernel_size
    sh, sw = stride
    top, bottom, left, right = padding
    oh = (h + top + bottom - kh) // sh + 1
    ow = (w + left + right - kw) // sw + 1
    if oh <= 0 or ow <= 0:
        raise ValueError("Convolution does not fit its input")
    group_size = 2 if readout == "pair_mean_hold" else 1
    groups = [
        tuple(range(start, min(start + group_size, ow)))
        for start in range(0, ow, group_size)
    ]
    tiles = []
    for phase in range(min(kw, sw)):
        phase_taps = len(range(phase, kw, sw))
        if phase_taps > kernel_slots or phase_taps > signal_slots:
            raise ValueError("Kernel phase does not fit its aperture")
        i = 0
        while i < len(groups):
            origin = groups[i][0]
            end = i
            while (
                end < len(groups)
                and groups[end][-1] - origin + phase_taps <= signal_slots
            ):
                end += 1
            if end == i:
                raise ValueError("Signal aperture cannot fit an entire readout group")
            local = tuple(
                tuple(kernel_slots - 1 + position - origin for position in group)
                for group in groups[i:end]
            )
            tiles.append(
                LinearReadoutTile(
                    phase,
                    phase_taps,
                    origin * sw + phase,
                    sw,
                    groups[end - 1][-1] - origin + phase_taps,
                    tuple(groups[i:end]),
                    local,
                )
            )
            i = end
    return LinearReadoutPlan(
        input_shape,
        (b, out_channels, oh, ow),
        kernel_size,
        stride,
        padding,
        nonnegative,
        readout,
        signal_slots,
        kernel_slots,
        dark_positions,
        tuple(tiles),
    )
