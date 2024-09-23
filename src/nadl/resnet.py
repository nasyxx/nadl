#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""Python ♡ Nasy.

    |             *         *
    |                  .                .
    |           .                              登
    |     *                      ,
    |                   .                      至
    |
    |                               *          恖
    |          |\___/|
    |          )    -(             .           聖 ·
    |         =\ -   /=
    |           )===(       *
    |          /   - \
    |          |-    |
    |         /   -   \     0.|.0
    |  NASY___\__( (__/_____(\=/)__+1s____________
    |  ______|____) )______|______|______|______|_
    |  ___|______( (____|______|______|______|____
    |  ______|____\_|______|______|______|______|_
    |  ___|______|______|______|______|______|____
    |  ______|______|______|______|______|______|_
    |  ___|______|______|______|______|______|____

author   : Nasy https://nasy.moe
date     : Sep 21, 2024
email    : Nasy <nasyxx+python@gmail.com>
filename : resnet.py
project  : nadl
license  : GPL-3.0+

ResNet follow resnet in torchvision.
"""

from collections.abc import Callable

import jax
from equinox import Module, field
from equinox.nn import (
  AdaptiveAvgPool2d,
  BatchNorm,
  Conv2d,
  GroupNorm,
  Linear,
  MaxPool2d,
  Sequential,
  State,
)

from jaxtyping import Array, PRNGKeyArray, PyTree
from typing import Protocol, Self

from .surgery import init_fn, init_surgery


def conv3x3(
  in_planes: int,
  out_planes: int,
  stride: int = 1,
  groups: int = 1,
  dilation: int = 1,
  *,
  key: PRNGKeyArray,
) -> Conv2d:
  """3x3 convolution with padding."""
  return Conv2d(
    in_planes,
    out_planes,
    kernel_size=3,
    stride=stride,
    padding=dilation,
    groups=groups,
    dilation=dilation,
    use_bias=False,
    key=key,
  )


def conv1x1(
  in_planes: int,
  out_planes: int,
  stride: int = 1,
  *,
  key: PRNGKeyArray,
) -> Conv2d:
  """1x1 convolution."""
  return Conv2d(
    in_planes,
    out_planes,
    kernel_size=1,
    stride=stride,
    use_bias=False,
    key=key,
  )


class Norm[**P](Protocol):
  """Norm layer."""

  weight: Array | None
  bias: Array | None

  def __init__(self, *args: P.args, **kwds: P.kwargs) -> None:
    """Initialize."""
    ...

  def __call__(
    self,
    x: Array,
    state: State,
    *,
    key: PRNGKeyArray | None = None,
    inference: bool | None = None,
  ) -> tuple[Array, State]:
    """Forward."""
    ...


class BasicBlock(Module):
  """Basic Block."""

  conv1: Conv2d
  bn1: Norm
  conv2: Conv2d
  bn2: Norm
  downsample: Callable[..., Array] | None
  stride: int = field(static=True)
  activation: Callable[[Array], Array]
  expansion: int = field(default=1, static=True)

  @classmethod
  def init(  # noqa: PLR0913
    cls,
    inplanes: int,
    planes: int,
    stride: int = 1,
    downsample: Callable | None = None,
    groups: int = 1,
    base_width: int = 64,
    dilation: int = 1,
    norm_layer: type[Norm] | None = None,
    activation: Callable[[Array], Array] = jax.nn.relu,
    expansion: int = 1,
    *,
    key: PRNGKeyArray,
  ) -> Self:
    """Initialize."""
    if norm_layer is None:
      norm_layer = BatchNorm
    assert groups == 1, "BasicBlock only supports groups=1"
    assert base_width == 64, "BasicBlock only supports base_width=64"  # noqa: PLR2004
    assert dilation <= 1, "BasicBlock only supports dilation <= 1"

    k1, k2 = jax.random.split(key)
    conv1 = conv3x3(inplanes, planes, stride, key=k1)
    bn1 = norm_layer(planes, axis_name="batch")
    conv2 = conv3x3(planes, planes, key=k2)
    bn2 = norm_layer(planes, axis_name="batch")

    return cls(conv1, bn1, conv2, bn2, downsample, stride, activation, expansion)

  def __call__(
    self, x: Array, state: State, *, key: PRNGKeyArray | None = None
  ) -> tuple[Array, State]:
    """Forward."""
    del key
    identity = x

    out = self.conv1(x)
    out, state = self.bn1(out, state)
    out = self.activation(out)

    out = self.conv2(out)
    out, state = self.bn2(out, state)

    if self.downsample is not None:
      identity, state = self.downsample(x, state)

    out += identity
    out = self.activation(out)

    return out, state


class Bottleneck(Module):
  """Bottleneck."""

  conv1: Conv2d
  bn1: Norm
  conv2: Conv2d
  bn2: Norm
  conv3: Conv2d
  bn3: Norm
  downsample: Callable[..., Array] | None
  stride: int = field(static=True)
  activation: Callable[[Array], Array]
  expansion: int = field(default=4, static=True)

  @classmethod
  def init(  # noqa: PLR0913
    cls,
    inplanes: int,
    planes: int,
    stride: int = 1,
    downsample: Callable | None = None,
    groups: int = 1,
    base_width: int = 64,
    dilation: int = 1,
    norm_layer: type[Norm] | None = None,
    activation: Callable[[Array], Array] = jax.nn.relu,
    expansion: int = 4,
    *,
    key: PRNGKeyArray,
  ) -> Self:
    """Initialize."""
    if norm_layer is None:
      norm_layer = BatchNorm
    width = int(planes * (base_width / 64.0)) * groups

    k1, k2, k3 = jax.random.split(key, 3)
    # Both self.conv2 and self.downsample layers downsample the input when stride != 1
    conv1 = conv1x1(inplanes, width, key=k1)
    bn1 = norm_layer(width, axis_name="batch")
    conv2 = conv3x3(width, width, stride, groups, dilation, key=k2)
    bn2 = norm_layer(width, axis_name="batch")
    conv3 = conv1x1(width, planes * expansion, key=k3)
    bn3 = norm_layer(planes * 4, axis_name="batch")

    return cls(
      conv1, bn1, conv2, bn2, conv3, bn3, downsample, stride, activation, expansion
    )

  def __call__(
    self, x: Array, state: State, *, key: PRNGKeyArray | None = None
  ) -> tuple[Array, State]:
    """Forward."""
    del key
    identity = x

    out = self.conv1(x)
    out, state = self.bn1(out, state)
    out = self.activation(out)

    out = self.conv2(out)
    out, state = self.bn2(out, state)
    out = self.activation(out)

    out = self.conv3(out)
    out, state = self.bn3(out, state)

    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    out = self.activation(out)
    return out, state


class StateSeq(Module):
  """Stateful sequential layers."""

  layers: list

  def __call__(
    self, x: Array, state: State, *, key: PRNGKeyArray | None = None
  ) -> tuple[Array, State]:
    """Forward."""
    ks = (
      jax.random.split(key, len(self.layers))
      if key is not None
      else (None,) * len(self.layers)
    )
    for layer, k in zip(self.layers, ks):
      x, state = layer(x, state, key=k)
    return x, state

  def __len__(self) -> int:
    """Length."""
    return len(self.layers)

  def __iter__(self):  # noqa: ANN204
    """Iter."""
    yield from self.layers


class ResNet(Module):
  """ResNet."""

  conv1: Conv2d
  bn1: BatchNorm | Callable
  maxpool: MaxPool2d
  layer1: StateSeq
  layer2: StateSeq
  layer3: StateSeq
  layer4: StateSeq
  avepool: AdaptiveAvgPool2d
  fc: Callable[..., Array]
  activation: Callable[[Array], Array]

  _make_layer: Callable

  inplanes: int = field(static=True)
  dilation: int = field(static=True)
  groups: int = field(static=True)
  base_width: int = field(static=True)

  def is_stateful(self) -> bool:  # noqa: PLR6301
    """Check if stateful."""
    return True

  @classmethod
  def init(  # noqa: PLR0914
    cls,
    block: type[BasicBlock | Bottleneck],
    layers: list[int],
    num_classes: int = 1000,
    zero_init_residual: bool = False,
    groups: int = 1,
    width_per_group: int = 64,
    replace_stride_with_dilation: list[bool] | None = None,
    norm_layer: type[Norm] | None = None,
    activation: Callable[[Array], Array] = jax.nn.relu,
    *,
    key: PRNGKeyArray,
  ) -> Self:
    """Initialize."""
    if norm_layer is None:
      norm_layer = BatchNorm
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    inplanes = 64
    dilation = 1
    if replace_stride_with_dilation is None:
      replace_stride_with_dilation = [False, False, False]
    assert len(replace_stride_with_dilation) == 3, (  # noqa: PLR2004
      "replace_stride_with_dilation should be None or a 3-element list, "
      f"got {replace_stride_with_dilation}"
    )
    base_width = width_per_group

    def _make_layer(
      block: type[BasicBlock | Bottleneck],
      planes: int,
      blocks: int,
      stride: int = 1,
      dilate: bool = False,
      inplanes: int = 64,
      dilation: int = dilation,
      *,
      key: PRNGKeyArray,
    ) -> tuple[StateSeq, int, int]:
      """Make layer."""
      k1, k2, *ks = jax.random.split(key, 1 + blocks)
      downsample = None
      preview_diation = dilation
      if dilate:
        dilation *= stride
        stride = 1
      if stride != 1 or inplanes != planes * block.expansion:
        downsample = Sequential([
          conv1x1(inplanes, planes * block.expansion, stride, key=k1),
          norm_layer(planes * block.expansion, axis_name="batch"),
        ])

      layers = [
        block.init(
          inplanes,
          planes,
          stride,
          downsample,
          groups,
          base_width,
          preview_diation,
          norm_layer,
          activation,
          key=k2,
        )
      ]
      inplanes = planes * block.expansion
      layers += list(
        map(
          lambda k: block.init(
            inplanes,
            planes,
            groups=groups,
            base_width=base_width,
            dilation=dilation,
            norm_layer=norm_layer,
            activation=activation,
            key=k,
          ),
          ks,
        )
      )
      return StateSeq(layers), inplanes, dilation

    conv1 = Conv2d(
      3, inplanes, kernel_size=7, stride=2, padding=3, use_bias=False, key=k1
    )
    bn1 = norm_layer(inplanes, axis_name="batch")
    maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)
    layer1, inplanes, dilation = _make_layer(
      block, 64, layers[0], inplanes=inplanes, dilation=dilation, key=k2
    )
    layer2, inplanes, dilation = _make_layer(
      block,
      128,
      layers[1],
      2,
      inplanes=inplanes,
      dilate=replace_stride_with_dilation[0],
      dilation=dilation,
      key=k3,
    )
    layer3, inplanes, dilation = _make_layer(
      block,
      256,
      layers[2],
      2,
      inplanes=inplanes,
      dilate=replace_stride_with_dilation[1],
      dilation=dilation,
      key=k4,
    )
    layer4, inplanes, dilation = _make_layer(
      block,
      512,
      layers[3],
      2,
      inplanes=inplanes,
      dilate=replace_stride_with_dilation[2],
      dilation=dilation,
      key=k5,
    )
    avepool = AdaptiveAvgPool2d((1, 1))
    fc = Linear(512 * block.expansion, num_classes, key=k5)

    model = cls(
      conv1,
      bn1,
      maxpool,
      layer1,
      layer2,
      layer3,
      layer4,
      avepool,
      fc,
      activation,
      _make_layer,
      inplanes,
      dilation,
      groups,
      base_width,
    )

    def conv2d_p(x: PyTree) -> bool:
      return isinstance(x, Conv2d)

    def bg_norm_p(x: PyTree) -> bool:
      return isinstance(x, BatchNorm | GroupNorm)

    def bg_norm_bias(x: BatchNorm | GroupNorm) -> Array | None:
      return x.bias

    model = init_surgery(model, k6, conv2d_p)
    model = init_surgery(model, k6, bg_norm_p, init_fn(jax.nn.initializers.constant(1)))
    model = init_surgery(
      model, k6, bg_norm_p, init_fn(jax.nn.initializers.constant(0)), bg_norm_bias
    )

    if zero_init_residual:

      def basicblock_p(x: PyTree) -> bool:
        return isinstance(x, BasicBlock) and x.bn2.weight is not None

      def bottleneck_p(x: PyTree) -> bool:
        return isinstance(x, Bottleneck) and x.bn3.weight is not None

      model = init_surgery(
        model,
        k6,
        basicblock_p,
        init_fn(jax.nn.initializers.constant(0)),
        get_weight=lambda x: x.bn2.weight,
      )

      model = init_surgery(
        model,
        k6,
        bottleneck_p,
        init_fn(jax.nn.initializers.constant(0)),
        get_weight=lambda x: x.bn3.weight,
      )
    return model

  def __call__(
    self, x: Array, state: State, *, key: PRNGKeyArray | None = None
  ) -> tuple[Array, State]:
    """Forward."""
    del key
    x = self.conv1(x)
    x, state = self.bn1(x, state)
    x = self.activation(x)
    x = self.maxpool(x)

    x, state = self.layer1(x, state)
    x, state = self.layer2(x, state)
    x, state = self.layer3(x, state)
    x, state = self.layer4(x, state)

    x = self.avepool(x)
    x = x.ravel()
    x = self.fc(x)
    return x, state


def resnet18(
  key: PRNGKeyArray,
  num_classes: int = 1000,
  zero_init_residual: bool = False,
  groups: int = 1,
  width_per_group: int = 64,
  replace_stride_with_dilation: list[bool] | None = None,
  norm_layer: type[Norm] | None = None,
  activation: Callable[[Array], Array] = jax.nn.relu,
) -> ResNet:
  """ResNet18."""
  return ResNet.init(
    BasicBlock,
    [2, 2, 2, 2],
    num_classes=num_classes,
    zero_init_residual=zero_init_residual,
    groups=groups,
    width_per_group=width_per_group,
    replace_stride_with_dilation=replace_stride_with_dilation,
    norm_layer=norm_layer,
    activation=activation,
    key=key,
  )


def resnet34(
  key: PRNGKeyArray,
  num_classes: int = 1000,
  zero_init_residual: bool = False,
  groups: int = 1,
  width_per_group: int = 64,
  replace_stride_with_dilation: list[bool] | None = None,
  norm_layer: type[Norm] | None = None,
  activation: Callable[[Array], Array] = jax.nn.relu,
) -> ResNet:
  """ResNet34."""
  return ResNet.init(
    BasicBlock,
    [3, 4, 6, 3],
    num_classes=num_classes,
    zero_init_residual=zero_init_residual,
    groups=groups,
    width_per_group=width_per_group,
    replace_stride_with_dilation=replace_stride_with_dilation,
    norm_layer=norm_layer,
    activation=activation,
    key=key,
  )


def resnet50(
  key: PRNGKeyArray,
  num_classes: int = 1000,
  zero_init_residual: bool = False,
  groups: int = 1,
  width_per_group: int = 64,
  replace_stride_with_dilation: list[bool] | None = None,
  norm_layer: type[Norm] | None = None,
  activation: Callable[[Array], Array] = jax.nn.relu,
) -> ResNet:
  """ResNet50."""
  return ResNet.init(
    Bottleneck,
    [3, 4, 6, 3],
    num_classes=num_classes,
    zero_init_residual=zero_init_residual,
    groups=groups,
    width_per_group=width_per_group,
    replace_stride_with_dilation=replace_stride_with_dilation,
    norm_layer=norm_layer,
    activation=activation,
    key=key,
  )


def resnet101(
  key: PRNGKeyArray,
  num_classes: int = 1000,
  zero_init_residual: bool = False,
  groups: int = 1,
  width_per_group: int = 64,
  replace_stride_with_dilation: list[bool] | None = None,
  norm_layer: type[Norm] | None = None,
  activation: Callable[[Array], Array] = jax.nn.relu,
) -> ResNet:
  """ResNet101."""
  return ResNet.init(
    Bottleneck,
    [3, 4, 23, 3],
    num_classes=num_classes,
    zero_init_residual=zero_init_residual,
    groups=groups,
    width_per_group=width_per_group,
    replace_stride_with_dilation=replace_stride_with_dilation,
    norm_layer=norm_layer,
    activation=activation,
    key=key,
  )


def resnet152(
  key: PRNGKeyArray,
  num_classes: int = 1000,
  zero_init_residual: bool = False,
  groups: int = 1,
  width_per_group: int = 64,
  replace_stride_with_dilation: list[bool] | None = None,
  norm_layer: type[Norm] | None = None,
  activation: Callable[[Array], Array] = jax.nn.relu,
) -> ResNet:
  """ResNet152."""
  return ResNet.init(
    Bottleneck,
    [3, 8, 36, 3],
    num_classes=num_classes,
    zero_init_residual=zero_init_residual,
    groups=groups,
    width_per_group=width_per_group,
    replace_stride_with_dilation=replace_stride_with_dilation,
    norm_layer=norm_layer,
    activation=activation,
    key=key,
  )


def resnext50_32x4d(
  key: PRNGKeyArray,
  num_classes: int = 1000,
  zero_init_residual: bool = False,
  groups: int = 32,
  width_per_group: int = 4,
  replace_stride_with_dilation: list[bool] | None = None,
  norm_layer: type[Norm] | None = None,
  activation: Callable[[Array], Array] = jax.nn.relu,
) -> ResNet:
  """ResNet50_32x4d."""
  return ResNet.init(
    Bottleneck,
    [3, 4, 6, 3],
    num_classes=num_classes,
    zero_init_residual=zero_init_residual,
    groups=groups,
    width_per_group=width_per_group,
    replace_stride_with_dilation=replace_stride_with_dilation,
    norm_layer=norm_layer,
    activation=activation,
    key=key,
  )


def resnext101_32x8d(
  key: PRNGKeyArray,
  num_classes: int = 1000,
  zero_init_residual: bool = False,
  groups: int = 32,
  width_per_group: int = 8,
  replace_stride_with_dilation: list[bool] | None = None,
  norm_layer: type[Norm] | None = None,
  activation: Callable[[Array], Array] = jax.nn.relu,
) -> ResNet:
  """ResNet101_32x8d."""
  return ResNet.init(
    Bottleneck,
    [3, 4, 23, 3],
    num_classes=num_classes,
    zero_init_residual=zero_init_residual,
    groups=groups,
    width_per_group=width_per_group,
    replace_stride_with_dilation=replace_stride_with_dilation,
    norm_layer=norm_layer,
    activation=activation,
    key=key,
  )


def resnext101_64x4d(
  key: PRNGKeyArray,
  num_classes: int = 1000,
  zero_init_residual: bool = False,
  groups: int = 64,
  width_per_group: int = 4,
  replace_stride_with_dilation: list[bool] | None = None,
  norm_layer: type[Norm] | None = None,
  activation: Callable[[Array], Array] = jax.nn.relu,
) -> ResNet:
  """ResNet101_64x4d."""
  return ResNet.init(
    Bottleneck,
    [3, 4, 23, 3],
    num_classes=num_classes,
    zero_init_residual=zero_init_residual,
    groups=groups,
    width_per_group=width_per_group,
    replace_stride_with_dilation=replace_stride_with_dilation,
    norm_layer=norm_layer,
    activation=activation,
    key=key,
  )


def wide_resnet50_2(
  key: PRNGKeyArray,
  num_classes: int = 1000,
  zero_init_residual: bool = False,
  groups: int = 1,
  width_per_group: int = 64,
  replace_stride_with_dilation: list[bool] | None = None,
  norm_layer: type[Norm] | None = None,
  activation: Callable[[Array], Array] = jax.nn.relu,
) -> ResNet:
  """Wide ResNet50_2."""
  return ResNet.init(
    Bottleneck,
    [3, 4, 6, 3],
    num_classes=num_classes,
    zero_init_residual=zero_init_residual,
    groups=groups,
    width_per_group=width_per_group,
    replace_stride_with_dilation=replace_stride_with_dilation,
    norm_layer=norm_layer,
    activation=activation,
    key=key,
  )


def wide_resnet101_2(
  key: PRNGKeyArray,
  num_classes: int = 1000,
  zero_init_residual: bool = False,
  groups: int = 1,
  width_per_group: int = 64,
  replace_stride_with_dilation: list[bool] | None = None,
  norm_layer: type[Norm] | None = None,
  activation: Callable[[Array], Array] = jax.nn.relu,
) -> ResNet:
  """Wide ResNet101_2."""
  return ResNet.init(
    Bottleneck,
    [3, 4, 23, 3],
    num_classes=num_classes,
    zero_init_residual=zero_init_residual,
    groups=groups,
    width_per_group=width_per_group,
    replace_stride_with_dilation=replace_stride_with_dilation,
    norm_layer=norm_layer,
    activation=activation,
    key=key,
  )
