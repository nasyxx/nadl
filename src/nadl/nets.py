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
date     : Dec  8, 2023
email    : Nasy <nasyxx+python@gmail.com>
filename : nets.py
project  : nadl
license  : GPL-3.0+

Pre define networks.
"""

from .blocks import DownSample, UpSample, ConvBlock, ResNetBasic, ResNetBottleneck

from flax import linen
import jax


class UNet(linen.Module):
  """UNet model for image segmentation."""

  out_channels: int
  num_features: int = 4
  init_features: int = 64
  kernel_size: int = 3
  use_depthwise: bool = False

  @linen.compact
  def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
    """Build the UNet model."""
    list_features = list(
      map(lambda n: self.init_features * (2**n), range(self.num_features))
    )
    skips = [
      x := DownSample(
        features=features,
        kernel_size=self.kernel_size,
        use_depthwise=self.use_depthwise,
        pool="before" if features > self.init_features else None,
      )(x, train=train)
      for features in list_features
    ]

    # Bottleneck
    x = ConvBlock(
      features=list_features[-1] * 2,
      kernel_size=self.kernel_size,
      use_depthwise=self.use_depthwise,
    )(
      linen.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID"),
      train=train,
    )

    # Upsample path
    for features in reversed(list_features):
      skip = skips.pop()
      x = UpSample(
        features=features,
        kernel_size=self.kernel_size,
        use_depthwise=self.use_depthwise,
      )(x, skip, train=train)

    # Final layer
    return linen.Conv(features=self.out_channels, kernel_size=(1, 1))(x)


def resent18(out_channels: int, init_features: int = 64) -> ResNetBasic:
  """ResNet18."""
  return ResNetBasic(
    out_channels=out_channels, init_features=init_features, name="ResNet18"
  )


def resnet34(out_channels: int, init_features: int = 64) -> ResNetBasic:
  """ResNet34."""
  return ResNetBasic(
    out_channels=out_channels,
    init_features=init_features,
    num_blocks=(3, 4, 6, 3),
    name="ResNet34",
  )


def resnet50(out_channels: int, init_features: int = 64) -> ResNetBottleneck:
  """ResNet50."""
  return ResNetBottleneck(
    out_channels=out_channels, init_features=init_features, name="ResNet50"
  )


def resnet101(out_channels: int, init_features: int = 64) -> ResNetBottleneck:
  """ResNet101."""
  return ResNetBottleneck(
    out_channels=out_channels,
    init_features=init_features,
    num_blocks=(3, 4, 23, 3),
    name="ResNet101",
  )


def reset152(out_channels: int, init_features: int = 64) -> ResNetBottleneck:
  """ResNet152."""
  return ResNetBottleneck(
    out_channels=out_channels,
    init_features=init_features,
    num_blocks=(3, 8, 36, 3),
    name="ResNet152",
  )
