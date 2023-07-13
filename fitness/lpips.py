r"""Learned Perceptual Image Patch Similarity (LPIPS)

This module implements the LPIPS in PyTorch.

Original:
    https://github.com/richzhang/PerceptualSimilarity
    
Modified from: 
    https://github.com/francois-rozet/piqa

References:
    .. [Zhang2018] The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (Zhang et al., 2018)

    .. [Deng2009] ImageNet: A large-scale hierarchical image database (Deng et al, 2009)
"""

import inspect
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torch.hub as hub

from torch import Tensor
from typing import Dict, List, Tuple

ORIGIN: str = 'https://github.com/richzhang/PerceptualSimilarity'
SHIFT: Tensor = torch.tensor([0.485, 0.456, 0.406])
SCALE: Tensor = torch.tensor([0.229, 0.224, 0.225])

def broadcastable(*shapes) -> bool:
    try:
        torch.broadcast_shapes(*shapes)
    except RuntimeError as e:
        return False
    else:
        return True

def assert_type(
    *tensors,
    device: torch.device = None,
    dim_range: Tuple[int, int] = None,
    n_channels: int = None,
    value_range: Tuple[float, float] = None,
) -> None:
    r"""Asserts that the types, devices, shapes and values of `tensors` are
    valid with respect to some requirements.
    Example:
        >>> x = torch.rand(5, 3, 256, 256)
        >>> y = torch.rand(5, 3, 256, 256)
        >>> assert_type(x, y, dim_range=(4, 4), n_channels=3)
    """

    if not __debug__:
        return

    if device is None:
        device = tensors[0].device

    shapes = [tuple(t.shape) for t in tensors]
    assert broadcastable(*shapes), f"Expected all tensors to have broadcastable shapes, but got {shapes}."

    for t in tensors:
        assert t.device == device, f"Expected all tensors to be on the same device, but got {str(t.device)} and {str(device)}."

        if dim_range is None:
            pass
        elif dim_range[0] == dim_range[1]:
            assert t.dim() == dim_range[0], f"Expected number of dimensions to be ' {dim_range[0]}, but got {t.dim()}."
        elif dim_range[0] < dim_range[1]:
            assert dim_range[0] <= t.dim(), f"Expected number of dimensions to be greater or equal to {dim_range[0]}, but got {t.dim()}."
            assert t.dim() <= dim_range[1], f"Expected number of dimensions to be lower or equal to {dim_range[1]}, but got {t.dim()}."
        else:
            assert dim_range[0] <= t.dim(), f"Expected number of dimensions to be greater or equal to {dim_range[0]}, but got {t.dim()}."

        if n_channels is not None:
            assert t.size(1) == n_channels, f"Expected number of channels to be {n_channels}, but got {t.size(1)}."

        if value_range is not None:
            assert value_range[0] <= t.min(), f"Expected all values to be greater or equal to {value_range[0]}, but got {t.min().item()}."
            assert t.max() <= value_range[1], f"Expected all values to be lower or equal to {value_range[1]}, but got {t.max().item()}."


def reduce_tensor(x: Tensor, reduction: str = 'mean') -> Tensor:
    if reduction == 'mean':
        return x.mean()
    elif reduction == 'sum':
        return x.sum()

    return x

def l2_norm(
    x: torch.Tensor,
    dims: List[int],
    keepdim: bool = False,
) -> torch.Tensor:
    x = x ** 2
    x = x.sum(dim=dims, keepdim=keepdim)
    x = x.sqrt()

    return x





def get_weights(
    network: str = 'alex',
    version: str = 'v0.1',
) -> Dict[str, Tensor]:
    r"""Returns the official LPIPS weights for `network`.

    Args:
        network: Specifies the perception network that is used:
            `'alex'` | `'squeeze'` | `'vgg'`.
        version: Specifies the official version release:
            `'v0.0'` | `'v0.1'`.

    Example:
        >>> w = get_weights(network='alex')
        >>> w.keys()
        dict_keys(['0.1.weight', '1.1.weight', '2.1.weight', '3.1.weight', '4.1.weight'])
    """

    # Load from URL
    weights = hub.load_state_dict_from_url(
        f'{ORIGIN}/raw/master/lpips/weights/{version}/{network}.pth',
        map_location='cpu',
    )

    # Format keys
    weights = {
        k.replace('lin', '').replace('.model', ''): v
        for (k, v) in weights.items()
    }

    return weights


class Intermediary(nn.Module):
    r"""Module that catches and returns the outputs of intermediate
    target layers of a sequential module during its forward pass.

    Args:
        layers: A sequential module.
        targets: A list of target layer indexes.
    """

    def __init__(self, layers: nn.Sequential, targets: List[int]):
        super().__init__()

        self.layers = nn.ModuleList()
        j = 0

        seq: List[nn.Module] = []

        for i, layer in enumerate(layers):
            seq.append(layer)

            if i == targets[j]:
                self.layers.append(nn.Sequential(*seq))
                seq.clear()

                j += 1
                if j == len(targets):
                    break

    def forward(self, input: Tensor) -> List[Tensor]:
        output = []

        for layer in self.layers:
            input = layer(input)
            output.append(input)

        return output


class LPIPS(nn.Module):
    r"""Creates a criterion that measures the LPIPS
    between an input :math:`x` and a target :math:`y`.

    .. math::
        \text{LPIPS}(x, y) = \sum_{l \, \in \, \mathcal{F}}
            w_l \cdot \text{MSE}(\phi_l(x), \phi_l(y))

    where :math:`\phi_l` represents the normalized output of an intermediate
    layer :math:`l` in a perceptual network :math:`\mathcal{F}`.

    Note:
        :class:`LPIPS` is a trainable metric. For more details, refer to [Zhang2018]_.

    Args:
        network: Specifies the perceptual network :math:`\mathcal{F}` to use:
            `'alex'` | `'squeeze'` | `'vgg'`.
        scaling: Whether the input and target need to be scaled w.r.t. [Deng2009]_.
        dropout: Whether dropout is used or not.
        pretrained: Whether the official weights :math:`w_l` are used or not.
        eval: Whether to initialize the object in evaluation mode or not.
        reduction: Specifies the reduction to apply to the output:
            `'none'` | `'mean'` | `'sum'`.

    Shapes:
        input: :math:`(N, 3, H, W)`
        target: :math:`(N, 3, H, W)`
        output: :math:`(N,)` or :math:`()` depending on `reduction`

    Example:
        >>> criterion = LPIPS().cuda()
        >>> x = torch.rand(5, 3, 256, 256, requires_grad=True).cuda()
        >>> y = torch.rand(5, 3, 256, 256).cuda()
        >>> l = criterion(x, y)
        >>> l.size()
        torch.Size([])
        >>> l.backward()
    """

    def __init__(
        self,
        vgg16_features,
        scaling: bool = True,
        dropout: bool = False,
        pretrained: bool = True,
        eval: bool = True,
        reduction: str = 'mean',
    ):
        super().__init__()

        # ImageNet scaling
        self.scaling = scaling
        self.register_buffer('shift', SHIFT.view(1, -1, 1, 1))
        self.register_buffer('scale', SCALE.view(1, -1, 1, 1))

        # Perception layers
        # if network == 'alex':  # AlexNet
        #     layers = models.alexnet(pretrained=True).features
        #     targets = [1, 4, 7, 9, 11]
        #     channels = [64, 192, 384, 256, 256]
        # elif network == 'squeeze':  # SqueezeNet
        #     layers = models.squeezenet1_1(pretrained=True).features
        #     targets = [1, 4, 7, 9, 10, 11, 12]
        #     channels = [64, 128, 256, 384, 384, 512, 512]
        # elif network == 'vgg':  # VGG16
        #     layers = models.vgg16(pretrained=True).features
        #     targets = [3, 8, 15, 22, 29]
        #     channels = [64, 128, 256, 512, 512]
        # else:
        #     raise ValueError(f'Unknown network architecture {network}')

        layers = vgg16_features
        targets = [3, 8, 15, 22, 29]
        channels = [64, 128, 256, 512, 512]
        

        self.net = Intermediary(layers, targets)
        for p in self.net.parameters():
            p.requires_grad = False

        # Linear comparators
        self.lins = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(inplace=True) if dropout else nn.Identity(),
                nn.Conv2d(c, 1, kernel_size=1, bias=False),
            ) for c in channels
        ])

        if pretrained:
            self.lins.load_state_dict(get_weights(network='vgg'))

        if eval:
            self.eval()

        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert_type(
            input, target,
            device=self.shift.device,
            dim_range=(4, 4),
            n_channels=3,
            value_range=(0., 1.) if self.scaling else (0., -1.),
        )

        # ImageNet scaling
        if self.scaling:
            input = (input - self.shift) / self.scale
            target = (target - self.shift) / self.scale

        # LPIPS
        residuals = []

        for lin, fx, fy in zip(self.lins, self.net(input), self.net(target)):
            fx = fx / l2_norm(fx, dims=[1], keepdim=True)
            fy = fy / l2_norm(fy, dims=[1], keepdim=True)

            mse = ((fx - fy) ** 2).mean(dim=(-1, -2), keepdim=True)
            residuals.append(lin(mse).flatten())

        l = torch.stack(residuals, dim=-1).sum(dim=-1)

        return reduce_tensor(l, self.reduction)
