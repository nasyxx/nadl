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
filename : blocks.py
project  : nadl
license  : GPL-3.0+

Layer Blocks
"""

from collections.abc import Sequence
from functools import partial

import jax
import jax.numpy as jnp
from flax import linen

from typing import Literal


class ConvBlock(linen.Module):
  """Define a convolutional block with optional depthwise conv and residual conn."""

  features: int
  kernel_size: int = 3
  use_depthwise: bool = True

  @linen.compact
  def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
    """Apply the convolution block to the input."""
    ks = (self.kernel_size, self.kernel_size)
    if self.use_depthwise:
      x = linen.Conv(
        features=x.shape[-1],
        kernel_size=ks,
        padding="SAME",
        feature_group_count=x.shape[-1],
      )(x)
      x = linen.Conv(features=self.features, kernel_size=(1, 1))(x)
    else:
      x = linen.Conv(self.features, ks, padding="SAME")(x)

    x = linen.BatchNorm()(x, use_running_average=not train)
    return linen.PReLU()(x)


class DownSample(linen.Module):
  """Downsample block for UNet encoder."""

  features: int
  kernel_size: int = 3
  use_depthwise: bool = False
  pool: Literal["before", "after", None] = "after"
  pool_type: Literal["max", "avg"] = "max"

  @linen.compact
  def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
    """Apply downsampling to the input."""
    match self.pool_type:
      case "max":
        pool_fn = linen.max_pool
      case "avg":
        pool_fn = linen.avg_pool
      case _:
        raise ValueError(f"Unknown pool type: {self.pool_type}")
    pool_fn = partial(pool_fn, window_shape=(2, 2), strides=(2, 2), padding="VALID")
    if self.pool == "before":
      x = pool_fn(x)

    x = ConvBlock(
      features=self.features,
      kernel_size=self.kernel_size,
      use_depthwise=self.use_depthwise,
    )(x, train=train)

    if self.pool == "after":
      x = pool_fn(x)
    return x


class UpSample(linen.Module):
  """Upsample block for UNet decoder."""

  features: int
  kernel_size: int = 3
  use_depthwise: bool = False

  @linen.compact
  def __call__(self, x: jax.Array, skip: jax.Array, train: bool = False) -> jax.Array:
    """Apply upsampling to the input and concatenate with skip connection."""
    x = linen.ConvTranspose(
      features=self.features,
      kernel_size=(2, 2),
      strides=(2, 2),
    )(x)

    if x.shape != skip.shape:
      diff = skip.shape[1] - x.shape[1]
      x = jnp.pad(
        x,
        [
          (0, 0),
          (diff // 2, diff - diff // 2),
          (diff // 2, diff - diff // 2),
          (0, 0),
        ],
      )

    x = jnp.concatenate([x, skip], axis=-1)
    return ConvBlock(
      features=self.features,
      kernel_size=self.kernel_size,
      use_depthwise=self.use_depthwise,
    )(x, train=train)


class BasicBlock(linen.Module):
  """Define a basic residual block."""

  features: int
  strides: tuple = (1, 1)
  use_projection: bool = False

  @linen.compact
  def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
    """Apply the basic residual block to the input."""
    identity = x

    # First convolutional layer
    y = linen.Conv(
      self.features, kernel_size=(3, 3), strides=self.strides, padding="SAME"
    )(x)
    y = linen.BatchNorm()(y, use_running_average=not train)
    y = linen.relu(y)

    # Second convolutional layer
    y = linen.Conv(self.features, kernel_size=(3, 3), padding="SAME")(y)
    y = linen.BatchNorm()(y, use_running_average=not train)

    # Projection shortcut, if required
    if self.use_projection:
      identity = linen.Conv(
        self.features, kernel_size=(1, 1), strides=self.strides, padding="SAME"
      )(x)
      identity = linen.BatchNorm()(identity, use_running_average=not train)

    # Adding the shortcut
    y += identity
    return linen.relu(y)


class BottleneckBlock(linen.Module):
  """Define a bottleneck residual block."""

  features: int
  expansion: int = 4
  strides: tuple = (1, 1)
  use_projection: bool = False

  @linen.compact
  def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
    """Apply the bottleneck residual block to the input."""
    identity = x

    # First layer
    y = linen.Conv(self.features, kernel_size=(1, 1))(x)
    y = linen.BatchNorm()(y, use_running_average=not train)
    y = linen.relu(y)

    # Second layer
    y = linen.Conv(
      self.features, kernel_size=(3, 3), strides=self.strides, padding="SAME"
    )(y)
    y = linen.BatchNorm()(y, use_running_average=not train)
    y = linen.relu(y)

    # Third layer
    y = linen.Conv(self.features * self.expansion, kernel_size=(1, 1))(y)
    y = linen.BatchNorm()(y, use_running_average=not train)

    # Projection shortcut
    if self.use_projection:
      identity = linen.Conv(
        self.features * self.expansion, kernel_size=(1, 1), strides=self.strides
      )(x)
      identity = linen.BatchNorm()(identity, use_running_average=not train)

    y += identity
    return linen.relu(y)


def make_layer(
  block: type[linen.Module],
  features: int,
  num_blocks: int,
  strides: tuple[int, int],
  use_projection: bool = False,
) -> list[linen.Module]:
  """Make a layer with the given block."""
  return list(
    map(
      lambda i: block(
        features,
        strides=strides if i == 0 else (1, 1),
        use_projection=use_projection if i == 0 else False,
      ),
      range(num_blocks),
    )
  )


class ResNetBasic(linen.Module):
  """Define a ResNet w/ Basic Block."""

  out_channels: int
  num_blocks: Sequence[int] = (2, 2, 2, 2)  # 2,2,2,2 -> ResNet 18
  init_features: int = 64

  @linen.compact
  def __call__(self, x: jax.Array, train: bool = False) -> tuple[jax.Array, jax.Array]:
    """Apply the ResNet18 to the input."""
    x = linen.Conv(
      self.init_features, kernel_size=(7, 7), strides=(2, 2), padding="SAME"
    )(x)
    x = linen.BatchNorm()(x, use_running_average=not train)
    x = linen.relu(x)
    x = linen.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")

    for i, bn in enumerate(self.num_blocks):
      x = linen.Sequential(
        make_layer(
          BasicBlock, self.init_features * (2**i), bn, (2, 2), use_projection=i > 0
        )
      )(x, train=train)

    res = x
    x = linen.avg_pool(x, window_shape=(x.shape[1], x.shape[2]))
    x = linen.Dense(self.out_channels)(x)
    return x, res


class ResNetBottleneck(linen.Module):
  """Define a ResNet w/ Bottleneck Block."""

  out_channels: int
  num_blocks: Sequence[int] = (3, 4, 6, 3)  # 3,4,6,3 -> ResNet 50
  init_features: int = 64

  @linen.compact
  def __call__(self, x: jax.Array, train: bool = False) -> tuple[jax.Array, jax.Array]:
    """Apply the ResNet50 to the input."""
    x = linen.Conv(
      self.init_features, kernel_size=(7, 7), strides=(2, 2), padding="SAME"
    )(x)
    x = linen.BatchNorm()(x, use_running_average=not train)
    x = linen.relu(x)
    x = linen.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")

    for i, bn in enumerate(self.num_blocks):
      x = linen.Sequential(
        make_layer(
          BottleneckBlock, self.init_features * (2**i), bn, (2, 2), use_projection=i > 0
        )
      )(x, train=train)

    res = x
    x = linen.avg_pool(x, window_shape=(x.shape[1], x.shape[2]))
    x = linen.Dense(self.out_channels)(x)
    return x, res
