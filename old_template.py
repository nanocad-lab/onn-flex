# -*- coding: utf-8 -*-
import math

# import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import init
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
import torchvision
import torchvision.transforms as transforms

# from torchvision.transforms import Compose
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
import random
# from scipy.io import loadmat

torch._C._jit_set_profiling_executor(False)

device = torch.device("cuda:0")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision('high')


def set_seed(seed_value):
	os.environ['PYTHONHASHSEED'] = str(seed_value)
	torch.manual_seed(seed_value)
	torch.cuda.manual_seed(seed_value)
	torch.cuda.manual_seed_all(seed_value)
	np.random.seed(seed_value)
	random.seed(seed_value)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

set_seed(0)

class PIC(nn.Module):
	def __init__(self, plane_size, sep):
		super(PIC, self).__init__()
		self.plane_size = plane_size
		self.sep = sep
	   
	def _perform_jtc_correlation_batch(self, signal_batch: torch.Tensor, kernel_batch: torch.Tensor) -> torch.Tensor:
		"""
		Performs the JTC-based correlation for a batch of 1D signals and kernels
		and returns the "same" sized portion of the correlation.

		Args:
			signal_batch (torch.Tensor): Batch of signals, shape [B, M].
										 M must be 8 for this implementation.
			kernel_batch (torch.Tensor): Batch of kernels, shape [B, N].
										 N must be 8 for this implementation.

		Returns:
			torch.Tensor: Batch of "same" sized correlation results, shape [B, M].
		"""
		B = signal_batch.shape[0] # Batch size
		M = signal_batch.shape[-1] # Length of signal
		N = kernel_batch.shape[-1] # Length of kernel

		if M != 8 or N != 8:
			raise ValueError(f"This JTC implementation expects signal and kernel lengths of 8. Got M={M}, N={N}.")

		kernel_complex_batch = kernel_batch.to(torch.complex64)
		signal_complex_batch = signal_batch.to(torch.complex64)

		plane_size = self.plane_size
		sep = self.sep

		input_plane_batch = torch.zeros(B, plane_size, dtype=torch.complex64, device=signal_batch.device)

		kernel_start = 0
		kernel_end = kernel_start + M
		signal_start = kernel_end + sep
		signal_end = signal_start + N

		input_plane_batch[:, kernel_start:kernel_end] = kernel_complex_batch
		input_plane_batch[:, signal_start:signal_end] = signal_complex_batch
		
		roll_amount = (plane_size // 2) - (M + signal_start) // 2 # 12
		
		input_plane_batch_rolled = torch.roll(input_plane_batch, shifts=roll_amount, dims=-1)

		jft_batch = torch.fft.fft(input_plane_batch_rolled, dim=-1)
		jft_batch_shifted = torch.fft.fftshift(jft_batch, dim=-1)

		jps_batch = torch.abs(jft_batch_shifted)**2
		jps_batch = jps_batch / plane_size

		output_plane_fft_batch = torch.fft.fft(jps_batch, dim=-1)
		output_plane_shifted_batch = torch.fft.fftshift(output_plane_fft_batch, dim=-1)
		
		output_plane_abs_batch = torch.abs(output_plane_shifted_batch)

		same_indices = (
			torch.arange(
				plane_size // 2 + sep + N // 2 + 1, plane_size // 2 + sep + N // 2 + 1 + 8
			)
			% plane_size
		)
		correlation_result_same_batch = output_plane_abs_batch[:, same_indices]

		return correlation_result_same_batch


	def forward(self, input, weights):
		# print(f"input shape: {input.shape}")
		# print(f"weights shape: {weights.shape}")
		ins = input.shape
		wes = weights.shape
		input_full = input.repeat(1, 1, wes[0], 1)
		weight_full = weights.repeat(ins[0], ins[1], 1, 1)

		# print(f"input_full shape: {input_full.shape}")
		# print(f"weight_full shape: {weight_full.shape}")

		orig_shape_dim0 = input_full.shape[0] # 128
		orig_shape_dim1 = input_full.shape[1] # 32
		orig_shape_dim2 = input_full.shape[2] # 8 (number of signals/kernels in this dimension)

		M = input_full.shape[-1] # Should be 8
		N = weight_full.shape[-1] # Should be 8

		if M != 8 or N != 8:
			raise ValueError(f"Input signal and kernel last dimension must be 8. Got {M} and {N}.")
		if input_full.shape[:-1] != weight_full.shape[:-1]:
			raise ValueError("Input signal and kernel_weights must have matching batch dimensions.")

		# Reshape inputs from [128, 32, 8, 8] to [X, 8] for batch processing
		# X = 128 * 32 * 8 (total number of 1D signals/kernels)
		batch_size_for_jtc = orig_shape_dim0 * orig_shape_dim1 * orig_shape_dim2
		
		signal_reshaped = input_full.reshape(batch_size_for_jtc, M)
		kernel_reshaped = weight_full.reshape(batch_size_for_jtc, N)

		# print(f"signal_reshaped shape: {signal_reshaped.shape}")
		# print(f"kernel_reshaped shape: {kernel_reshaped.shape}")

		# Perform JTC correlation for the batch
		# Output will be of shape [batch_size_for_jtc, L_conv]
		# L_conv = M + N - 1 = 8 + 8 - 1 = 15
		correlation_output_batched = self._perform_jtc_correlation_batch(signal_reshaped, kernel_reshaped)
		
		# Reshape the output back to the desired format [128, 32, 8, 15]
		output_reshaped = correlation_output_batched.reshape(orig_shape_dim0, 
															  orig_shape_dim1, 
															  orig_shape_dim2, 
															  N)
		
		# print(f"output_reshaped shape: {output_reshaped.shape}")
		#raise Exception("stop here")
		return output_reshaped


class _ConvNd(Module):
	__constants__ = [
		"stride",
		"padding",
		"dilation",
		"groups",
		"bias",
		"padding_mode",
		"output_padding",
		"in_channels",
		"out_channels",
		"kernel_size",
	]

	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		kernel_size: int,
		batch_size: int,
		stride: tuple[int, int],
		padding: tuple[int, int],
		dilation: tuple[int, int],
		transposed: bool,
		output_padding: tuple[int, int],
		groups: int,
		bias: bool,
		padding_mode: str,
	):
		super(_ConvNd, self).__init__()
		if in_channels % groups != 0:
			raise ValueError("in_channels must be divisible by groups")
		if out_channels % groups != 0:
			raise ValueError("out_channels must be divisible by groups")
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.batch_size = batch_size
		self.stride = stride
		self.padding = padding
		self.dilation = dilation
		self.transposed = transposed
		self.output_padding = output_padding
		self.groups = groups
		self.padding_mode = padding_mode
		self.weights = Parameter(
			torch.Tensor(in_channels, out_channels // groups, kernel_size, 2)
		)
		self.cout_per_cin = out_channels // groups
		if bias:
			self.bias = Parameter(torch.Tensor(out_channels))
		else:
			self.register_parameter("bias", None)
		self.reset_parameters()

	def reset_parameters(self):
		init.kaiming_uniform_(self.weights, a=math.sqrt(5))

		if self.bias is not None:
			fan_in, _ = init._calculate_fan_in_and_fan_out(self.weights)
			bound = 1 / math.sqrt(fan_in)
			init.uniform_(self.bias, -bound, bound)

	def extra_repr(self):
		s = (
			"{in_channels}, {out_channels}, kernel_size={kernel_size}"
			", stride={stride}"
		)
		if self.padding != (0,) * len(self.padding):
			s += ", padding={padding}"
		if self.dilation != (1,) * len(self.dilation):
			s += ", dilation={dilation}"
		if self.output_padding != (0,) * len(self.output_padding):
			s += ", output_padding={output_padding}"
		if self.groups != 1:
			s += ", groups={groups}"
		if self.bias is None:
			s += ", bias=False"
		return s.format(**self.__dict__)

	def __setstate__(self, state):
		super(_ConvNd, self).__setstate__(state)
		if not hasattr(self, "padding_mode"):
			self.padding_mode = "zeros"


class FTconvlayer(_ConvNd):
	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		kernel_size: int,
		batch_size: int = 128,
		stride: int = 1,
		padding: int = 0,
		dilation: int = 1,
		groups: int = 1,
		bias: bool = True,
		padding_mode: str = "zeros",
		vertical: bool = False,
		hv_concat: bool = False,
		plane_size: int = 48,
		sep: int = 8,
	):
		super(FTconvlayer, self).__init__(
			in_channels,
			out_channels,
			kernel_size,
			batch_size,
			stride,
			padding,
			dilation,
			False,
			(0, 0),
			groups,
			bias,
			padding_mode,
		)
		self.vertical = vertical
		self.hv_concat = hv_concat
		self.PIC_CONV = PIC(plane_size, sep)

	def hardware_forward(self, input, weight):
		return self.PIC_CONV(input, weight)

	def conv_forward(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
		# Original logic for kernel_size=8
		# (Unchanged from your original code)
		input_shape = input.shape
		# if input_shape[-1] == 28:
		#     # input layer of MNIST, pad to 32x32
		#     input = F.pad(input, (2, 2, 2, 2))
		w = input.shape[2]
		output = torch.zeros(
			input_shape[0], self.out_channels, w, w, device=input.device
		)
		# print("#"*60)
		# print(f"output shape: {output.shape}")
		# print(f"original input shape: {input.shape}")
		input = input.permute(0, 3, 1, 2)  # B, C, H, W to B, W, C, H??? to B, C, H,
		# print(f"input shape after permute: {input.shape}")
		# Process patches of size 8 along the spatial dimension
		for c_in in range(input.shape[2]):
			input_c = input[:, :, c_in : c_in + 1, ...]
			# print(f"input_c shape: {input_c.shape}")
			weight_c = weight[c_in, ...]
			# print(f"weight_c shape: {weight_c.shape}")
			n_patch = int(input.shape[3] / 8)
			# print(f"n_patch: {n_patch}")
			c_out_start = (c_in % self.groups) * self.cout_per_cin
			c_out_end = c_out_start + self.cout_per_cin
			# print(f"c_in: {c_in}, cout_per_cin: {self.cout_per_cin}, c_out_start: {c_out_start}, c_out_end: {c_out_end}")
			for i_p in range(n_patch):
				patch = input_c[..., 8 * i_p : 8 * i_p + 8]
				# print(f"patch shape: {patch.shape}")
				system_out = self.hardware_forward(patch, weight_c).permute(0, 2, 3, 1)
				# print(f"system_out permute shape: {system_out.shape}")
				# print(f"output patch shape: {output[:, c_out_start:c_out_end, 8 * i_p : 8 * i_p + 8, :].shape}")
				output[:, c_out_start:c_out_end, 8 * i_p : 8 * i_p + 8, :] += system_out
		return output

	def pseudo_forward(self, input, weight):
		# Wrapper function for pseudo-negative implementation
		weight_p = weight[..., 0]
		weight_n = weight[..., 1]  # pseudo-negative
		output_p = self.conv_forward(input, weight_p)
		output_n = self.conv_forward(input, weight_n)
		output = output_p - output_n
		return output

	def scale_signal(self, x: torch.Tensor) -> torch.Tensor:
		return x * (1 - self.Lowest) + self.Lowest

	def forward(self, input):

		if self.hv_concat:  # case for concat horizontal and vertical convs
			conv_tiled_h = self.pseudo_forward(input, self.weights)
			conv_tiled_v = self.pseudo_forward(
				input.permute(0, 1, 3, 2), self.weights
			).permute(0, 1, 3, 2)
			# conv_tiled_concat = torch.cat((conv_tiled_h, conv_tiled_v), 1)
			conv_tiled_stacked = torch.stack(
				[conv_tiled_h, conv_tiled_v], dim=2
			)  # Shape: (N, C, 2, H, W)
			conv_tiled_interleaved = conv_tiled_stacked.view(
				conv_tiled_h.size(0), -1, conv_tiled_h.size(2), conv_tiled_h.size(3)
			)  # Shape: (N, 2C, H, W)
			return conv_tiled_interleaved

		# case for horizontal and vertical convs
		if self.vertical:
			return self.pseudo_forward(input.permute(0, 1, 3, 2), self.weights).permute(
				0, 1, 3, 2
			)

		return self.pseudo_forward(input, self.weights)


batch_size = 128
stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
train_transform = transforms.Compose(
	[
		# transforms.RandomPerspective(distortion_scale=0.1, p=0.8),
		transforms.RandomHorizontalFlip(),
		# transforms.RandomVerticalFlip(),
		# transforms.RandomAffine(degrees=(-10, 10), scale=(0.90, 1.10), translate=(0.1, 0.1), shear=(-5, 5)),
		transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
		transforms.ToTensor(),
		transforms.Normalize(*stats, inplace=True),
	]
)
test_transform = transforms.Compose(
	[transforms.ToTensor(), transforms.Normalize(*stats)]
)
# train_transform = transforms.Compose([transforms.ToTensor()])
# test_transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(
	root="./data", train=True, download=True, transform=train_transform
)
trainloader = torch.utils.data.DataLoader(
	trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
)

testset = torchvision.datasets.CIFAR10(
	root="./data", train=False, download=True, transform=test_transform
)
testloader = torch.utils.data.DataLoader(
	testset,
	batch_size=batch_size,
	drop_last=True,
	shuffle=False,
	num_workers=4,
	pin_memory=True,
)

N_FILTER = 8
N_FILTER_adjust = N_FILTER * 2  # =2*N_FILTER if concat horizontal and vertical results


class FFTconv(nn.Module):
	def __init__(self, plane_size, sep):
		super(FFTconv, self).__init__()
		self.conv1 = FTconvlayer(3, 8, kernel_size=8, stride=1, hv_concat=True, plane_size=plane_size, sep=sep)
		self.bn1 = nn.BatchNorm2d(16)
		self.relu1 = nn.ReLU(inplace=True)

		self.maxpool1 = nn.MaxPool2d(2)

		self.conv2 = FTconvlayer(16, 16, kernel_size=8, stride=1, hv_concat=True, plane_size=plane_size, sep=sep)
		self.bn2 = nn.BatchNorm2d(32)
		self.relu2 = nn.ReLU(inplace=True)

		self.maxpool2 = nn.MaxPool2d(2)

		self.conv3 = FTconvlayer(32, 16, kernel_size=8, stride=1, hv_concat=True, plane_size=plane_size, sep=sep)
		self.bn3 = nn.BatchNorm2d(32)
		self.relu3 = nn.ReLU(inplace=True)

		self.conv4 = FTconvlayer(32, 16, kernel_size=8, stride=1, hv_concat=True, plane_size=plane_size, sep=sep)
		self.bn4 = nn.BatchNorm2d(32)
		self.relu4 = nn.ReLU(inplace=True)

		self.conv5 = FTconvlayer(32, 16, kernel_size=8, stride=1, hv_concat=True, plane_size=plane_size, sep=sep)
		self.bn5 = nn.BatchNorm2d(32)
		self.relu5 = nn.ReLU(inplace=True)

		self.conv6 = FTconvlayer(32, 16, kernel_size=8, stride=1, hv_concat=True, plane_size=plane_size, sep=sep)
		self.bn6 = nn.BatchNorm2d(32)
		self.relu6 = nn.ReLU(inplace=True)

		self.conv7 = FTconvlayer(32, 16, kernel_size=8, stride=1, hv_concat=True, plane_size=plane_size, sep=sep)
		self.bn7 = nn.BatchNorm2d(32)
		self.relu7 = nn.ReLU(inplace=True)

		self.classifier = nn.Sequential(
			nn.MaxPool2d(2), nn.Flatten(), nn.Linear(512, 256), nn.Linear(256, 10)
		)

	def forward(self, xb):
		out = self.conv1(xb)
		out = self.bn1(out)
		out = self.maxpool1(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.maxpool2(out)

		out = self.conv3(out)
		out = self.bn3(out)
		out = self.relu3(out)

		out = self.conv4(out)
		out = self.bn4(out)
		out = self.relu4(out)

		out = self.conv5(out)
		out = self.bn5(out)
		out = self.relu5(out)

		out = self.conv6(out)
		out = self.bn6(out)
		out = self.relu6(out)

		out = self.conv7(out)
		out = self.bn7(out)
		out = self.relu7(out)

		out = self.classifier(out)
		return out

	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features


def get_train_acc(net):
	net.eval()
	correct = 0
	total = 0
	with torch.no_grad():
		for data in trainloader:
			images, labels = data
			images, labels = images.to(device), labels.to(device)
			outputs = net(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	return 100 * correct / total


def get_test_acc(net):
	net.eval()
	correct = 0
	total = 0
	with torch.no_grad():
		for data in testloader:
			images, labels = data
			images, labels = images.to(device), labels.to(device)
			outputs = net(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	return 100 * correct / total

def train_fftconv(plane_size, sep):
	set_seed(0)
	net = FFTconv(plane_size, sep).to(device)

	init_lr = 1e-3
	n_epoch = 20
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.AdamW(net.parameters(), lr=init_lr)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)
	# torch.autograd.set_detect_anomaly(False)

	best_prec1 = 0

	net.train()
	# pbar = tqdm(range(n_epoch), desc="Epochs")
	for epoch in range(n_epoch):  # loop over the dataset multiple times
		pbar_inner = tqdm(trainloader, desc=f"Epoch {epoch}")
		running_loss = 0.0
		for data in pbar_inner:
			# print("new batch starts")
			# get the inputs; data is a list of [inputs, labels]
			inputs, labels = data
			inputs, labels = inputs.to(device, non_blocking=True), labels.to(
				device, non_blocking=True
			)

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			running_loss += loss.item()
			loss.backward()
			optimizer.step()

			pbar_inner.set_postfix(
				{
					"loss": f"{loss.item():.3}",
				}
			)

		test_acc = get_test_acc(net)
		train_acc = get_train_acc(net)
		net.train()
		best_prec1 = max(test_acc, best_prec1)
		scheduler.step()
		# Update progress bar
		print(
			{
				"epoch": epoch,
				"train_acc": f"{train_acc:.1f}",
				"test_acc": f"{test_acc:.1f}",
				"best_acc": f"{best_prec1:.1f}",
				"loss": f"{(running_loss/len(trainloader)):.3}",
			}
		)
	

	return best_prec1

results_filename = "fftconv_nosq_7layer_results.txt"
with open(results_filename, "a") as f:
	print("plane_size, sep, best_acc", file=f)

plane_size_values = range(16, 52, 4)
sep_values = range(0, 8)
for i, plane_size in enumerate(plane_size_values):
	for j, sep in enumerate(sep_values):
		if plane_size < 24:
			continue
		if 2 * 8 + sep > plane_size:
			continue
		print(f"Starting training for Plane size: {plane_size}, Sep: {sep}")
		best_acc = train_fftconv(plane_size, sep)
		print(f"Training complete for Plane size: {plane_size}, Sep: {sep}, Best accuracy: {best_acc:.2f}%")
		with open(results_filename, "a") as f:
			print(f"{plane_size}, {sep}, {best_acc:.3f}", file=f)
