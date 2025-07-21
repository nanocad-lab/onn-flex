from onn_config import AppConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.common_types import _size_2_t  # type: ignore
from torch.nn.modules.utils import _pair
from typing import Union  # Added List for _pair
from onn_component import JTC

# Helper for ceiling division if needed, though not directly used in the final version
# def ceil_div(a, b):
#     return (a + b - 1) // b


class FFTConv2d(_ConvNd):
    """
    Custom Conv2d implementation that uses F.conv1d with a fixed kernel size
    and padding="same" as its core computation engine after unfolding.
    """

    def __init__(
        self,
        args: AppConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        hw_size: int = 16,  # The fixed kernel size for the internal Conv1D
        jtc: JTC = None,
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(FFTConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
        )
        self.args = args
        self.hw_size = hw_size

        # Initialize weights
        self.weight = nn.Parameter(
            torch.empty(
                (out_channels, in_channels // groups, *kernel_size, 2),
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        self.hardware_forward = self.jtc.forward

    def pseudo_forward(self, input, weight):
        # Wrapper function for pseudo-negative implementation
        weight_p = weight[..., 0]
        weight_n = weight[..., 1]  # pseudo-negative
        output_p = self.conv_forward(input, weight_p)
        output_n = self.conv_forward(input, weight_n)
        output = output_p - output_n
        return output

    def forward(self, input, weight):
        return self.pseudo_forward(input, weight)

    def conv_forward(self, input, weight):
        if self.args.conv_method == "patch":
            return self.patch_forward(input, weight)
        elif self.args.conv_method == "dot_product":
            return self.dot_product_forward(input, weight)
        elif self.args.conv_method == "tile":
            return self.tile_forward(input, weight)

    def patch_forward(self, input, weight):
        """
        Patch-wise convolution using a hardware-accelerated 1D conv. (from Shurui)
        """
        input_shape = input.shape
        w = input.shape[2]
        output = torch.zeros(
            input_shape[0], self.out_channels, w, w, device=input.device
        )
        input = input.permute(0, 3, 1, 2)  # B, C, H, W to B, W, C, H??? to B, C, H,
        for c_in in range(input.shape[2]):
            input_c = input[:, :, c_in : c_in + 1, ...]
            weight_c = weight[c_in, ...]
            n_patch = int(input.shape[3] / 8)
            c_out_start = (c_in % self.groups) * self.cout_per_cin
            c_out_end = c_out_start + self.cout_per_cin
            for i_p in range(n_patch):
                patch = input_c[..., 8 * i_p : 8 * i_p + 8]
                system_out = self.hardware_forward(patch, weight_c)
                system_out = system_out.permute(0, 2, 3, 1)
                output[:, c_out_start:c_out_end, 8 * i_p : 8 * i_p + 8, :] += system_out
        return output

    def dot_product_forward(self, input, weight):
        pass

    def tile_forward(self, input, weight):
        """
        2D convolution via row-tiling and a hardware-accelerated 1D conv. (tiling_test_20250517.ipynb)
        """
        tile_h = 1
        N, C_in, H, W = input.shape
        C_out, _, kH, kW = weight.shape
        x_p = F.pad(
            input, (self.padding[0], self.padding[0], self.padding[1], self.padding[1])
        )
        H_p, W_p = H + 2 * self.padding[0], W + 2 * self.padding[1]
        H_out = (H_p - (kH - 1) * self.dilation[0] - 1) // self.stride[0] + 1
        W_out = (W_p - (kW - 1) * self.dilation[1] - 1) // self.stride[1] + 1

        out = input.new_zeros((N, C_out, H_out, W_out))
        for row_start in range(0, H_out, tile_h):
            th = min(tile_h, H_out - row_start)
            acc = input.new_zeros((N, C_out, th, W_out))
            for p in range(kH):
                slice_start = row_start + p * self.dilation[0]
                slice_p = x_p[
                    :,
                    :,
                    slice_start : slice_start + th,
                    :,
                ]
                batch_1d = slice_p.permute(0, 2, 1, 3).reshape(N * th, C_in, W_p)
                w_p = weight[:, :, p, :]
                seg_out = self.tiled_accelerator_conv1d(
                    batch_1d,
                    w_p,
                )
                out_tile = seg_out.view(N, th, C_out, W_out).permute(0, 2, 1, 3)
                acc += out_tile
            if self.bias is not None:
                acc += self.bias.view(1, -1, 1, 1)
            out[:, :, row_start : row_start + th, :] = acc
        return out

    def tiled_accelerator_conv1d(
        self, input: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        """
        1D convolution using a hardware accelerator that supports only
        fixed-length input segments <= hardware_size. The accelerator
        internally pads by hardware_size-1 and returns the full output.
        We crop the maximal valid outputs per segment, handling tail segments.
        """
        N, C_in, L_in = input.shape
        C_out, _, kW = weight.shape
        dilated_k = (kW - 1) * self.dilation[0] + 1
        if self.hw_size < dilated_k:
            raise ValueError(
                f"Hardware size ({self.hw_size}) < kernel size ({dilated_k})"
            )

        pad_hw = self.hw_size - 1
        step = self.hw_size - (dilated_k - 1)

        # Global pad
        input_p = F.pad(input, (self.padding[0], self.padding[0]))
        L_p = L_in + 2 * self.padding[0]
        L_out = (L_p - dilated_k) // self.stride[0] + 1
        out = input.new_zeros((N, C_out, L_out))

        for seg_start in range(0, L_p - dilated_k + 1, step):
            # Extract or pad segment to hardware_size
            seg = input_p[:, :, seg_start : seg_start + self.hw_size]
            if seg.size(2) < self.hw_size:
                seg = F.pad(seg, (0, self.hw_size - seg.size(2)))
            # Hardware conv: full padded-out
            seg_out_full = F.conv1d(
                seg,
                weight,
                bias=None,
                stride=self.stride[0],
                padding=pad_hw,
                dilation=self.dilation[0],
            )
            # Crop valid region
            seg_out = seg_out_full[:, :, pad_hw : pad_hw + step]
            out_start = seg_start // self.stride[0]
            # Handle tail overlap
            length = min(seg_out.size(2), L_out - out_start)
            out[:, :, out_start : out_start + length] = seg_out[:, :, :length]

        if self.bias is not None:
            out += self.bias.view(1, -1, 1)
        return out
