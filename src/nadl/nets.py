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

from collections.abc import Callable

import equinox as eqx
import jax

from jaxtyping import Array, Float, PRNGKeyArray

from .blocks import FastKANLayer
from .resnet import (
  resnet18,
  resnet34,
  resnet50,
  resnet101,
  resnet152,
  ResNet,
  resnext50_32x4d,
  resnext101_32x8d,
  resnext101_64x4d,
  wide_resnet50_2,
  wide_resnet101_2,
)


class pMTnet(eqx.Module):  # noqa: N801
  """pMTnet.

  https://github.com/tianshilu/pMTnet

  Deep learning neural network prediction tcr binding specificity to
  peptide and HLA based on peptide sequences. Please refer to our
  paper for more details: 'Deep learning-based prediction of T cell
  receptor-antigen binding specificity.

  (https://www.nature.com/articles/s42256-021-00383-2)

  Lu, T., Zhang, Z., Zhu, J. et al. 2021.
  """

  layers: eqx.nn.Sequential

  def __init__(
    self,
    inp: int,
    out: int = 1,
    hiddens: tuple[int, int, int] = (300, 200, 100),
    dropout_rate: float = 0.2,
    *,
    key: PRNGKeyArray,
  ) -> None:
    """Initialize the pMTnet."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    self.layers = eqx.nn.Sequential([
      eqx.nn.Linear(inp, hiddens[0], key=k1),
      eqx.nn.Dropout(p=dropout_rate),
      eqx.nn.Lambda(jax.nn.relu),
      eqx.nn.Linear(hiddens[0], hiddens[1], key=k2),
      eqx.nn.Lambda(jax.nn.relu),
      eqx.nn.Linear(hiddens[1], hiddens[2], key=k3),
      eqx.nn.Lambda(jax.nn.relu),
      eqx.nn.Linear(hiddens[2], out, key=k4),
    ])

  def __call__(self, x: Float[Array, " A"]) -> Float[Array, " A"]:
    """Forward."""
    return self.layers(x)


class FastKAN(eqx.Module):
  """FastKAN.

  FastKAN: Very Fast Implementation (Approximation) of Kolmogorov-Arnold Network.
  """

  layers: eqx.nn.Sequential

  def __init__(
    self,
    layers_hidden: list[int],
    grid_min: float = -2.0,
    grid_max: float = 2.0,
    num_grids: int = 8,
    use_base_update: bool = True,
    base_activation: Callable[[Float[Array, "..."]], Float[Array, "..."]] = jax.nn.silu,
    spline_weight_init_scale: float = 0.1,
    *,
    key: PRNGKeyArray,
  ) -> None:
    """Initialize the FastKAN."""
    ks = jax.random.split(key, len(layers_hidden) - 1)
    self.layers = eqx.nn.Sequential([
      FastKANLayer(
        in_dim,
        out_dim,
        grid_min=grid_min,
        grid_max=grid_max,
        num_grids=num_grids,
        use_base_update=use_base_update,
        base_activation=base_activation,
        spline_weight_init_scale=spline_weight_init_scale,
        key=k,
      )
      for in_dim, out_dim, k in zip(
        layers_hidden[:-1], layers_hidden[1:], ks, strict=True
      )
    ])

  def __call__(self, x: Float[Array, " A"]) -> Float[Array, " A"]:
    """Forward."""
    return self.layers(x)


__all__ = [
  "FastKAN",
  "FastKANLayer",
  "ResNet",
  "pMTnet",
  "resnet18",
  "resnet34",
  "resnet50",
  "resnet101",
  "resnet152",
  "resnext50_32x4d",
  "resnext101_32x8d",
  "resnext101_64x4d",
  "wide_resnet50_2",
  "wide_resnet101_2",
]
