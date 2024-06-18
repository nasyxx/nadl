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

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxtyping import Array, Float, PRNGKeyArray


class SplineLinear(eqx.Module):
  """Spline Linear layer."""

  init_scale: float
  liner: eqx.nn.Linear

  def __init__(
    self,
    in_features: int,
    out_features: int,
    init_scale: float = 1.0,
    use_bias: bool = True,
    *,
    key: PRNGKeyArray,
  ) -> None:
    """Initialize."""
    self.init_scale = init_scale
    linear = eqx.nn.Linear(in_features, out_features, use_bias, key=key)
    self.linear = eqx.tree_at(
      lambda ly: ly.weight,
      linear,
      jax.nn.initializers.truncated_normal(stddev=init_scale)(key, linear.weight.shape),
    )

  def __call__(self, x: Float[Array, " A"]) -> Float[Array, " A"]:
    """Forward."""
    return self.linear(x)


class RadialBasisFunction(eqx.Module):
  """Gaussian Radial Basis Function."""

  grid: Float[Array, " A"]
  num_grids: int
  grid_min: float
  grid_max: float
  denominator: float

  def __init__(
    self,
    grid_min: float = -2.0,
    grid_max: float = 2.0,
    num_grids: int = 5,
    denominator: float = 1,  # larger denominators lead to smoother basis
  ) -> None:
    """Initialize."""
    self.grid = jnp.linspace(grid_min, grid_max, num_grids)
    self.num_grids = num_grids
    self.grid_min = grid_min
    self.grid_max = grid_max
    self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

  def __call__(self, x: Float[Array, " A"]) -> Float[Array, " A"]:
    """Forward."""
    return jnp.exp(-(((x[..., None] - self.grid) / self.denominator) ** 2))


class FastKANLayer(eqx.Module):
  """Gaussian Radial Basis Functions replace the spline fast KAN.

  From https://github.com/ZiyaoLi/fast-kan/blob/master/fastkan/fastkan.py
  """

  rbf: RadialBasisFunction
  spline_linear: SplineLinear
  use_base_update: bool
  use_layernorm: bool
  layernorm: eqx.nn.LayerNorm | None = None
  base_activation: Callable[[Float[Array, "..."]], Float[Array, "..."]] = jax.nn.silu
  base_linear: eqx.nn.Linear | None = None

  def __init__(  # noqa: PLR0913
    self,
    input_dim: int,
    output_dim: int,
    grid_min: float = -2.0,
    grid_max: float = 2.0,
    num_grids: int = 5,
    use_base_update: bool = True,
    use_layernorm: bool = True,
    base_activation: Callable[[Float[Array, "..."]], Float[Array, "..."]] = jax.nn.silu,
    spline_weight_init_scale: float = 0.1,
    *,
    key: PRNGKeyArray,
  ) -> None:
    """Initialize."""
    self.use_layernorm = use_layernorm
    if use_layernorm:
      assert (
        input_dim > 1
      ), "Do not use layernorms on 1D inputs. Set `use_layernorm=False`."
      self.layernorm = eqx.nn.LayerNorm(input_dim)

    self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)

    self.spline_linear = SplineLinear(
      input_dim * num_grids, output_dim, spline_weight_init_scale, key=key
    )

    self.use_base_update = use_base_update
    if use_base_update:
      self.base_activation = base_activation
      self.base_linear = eqx.nn.Linear(input_dim, output_dim, key=key)

  def __call__(
    self, x: Float[Array, " A"]
  ) -> Float[Array, " A"]:
    """Forward."""
    if self.use_layernorm:
      assert self.layernorm is not None, "Layernorm is not initialized."
      x = self.layernorm(x)
    spline_basis = self.rbf(x)

    ret = self.spline_linear(spline_basis.reshape(*spline_basis.shape[:-2], -1))

    if self.use_base_update:
      assert self.base_linear is not None, "Base linear is not initialized."
      base = self.base_linear(self.base_activation(x))
      ret += base

    return ret

  @property
  def input_dim(self) -> int:
    """Get input dimension."""
    return self.spline_linear.linear.weight.shape[1] // self.rbf.num_grids

  @property
  def output_dim(self) -> int:
    """Get output dimension."""
    return self.spline_linear.linear.weight.shape[0]

  def plot_curve(
    self,
    input_index: int,
    output_index: int,
    num_pts: int = 1000,
    num_extrapolate_bins: int = 2,
  ) -> tuple[Float[Array, " A"], Float[Array, " A"]]:
    """This function returns the learned curves in a FastKANLayer.

    input_index: the selected index of the input, in [0, input_dim) .
    output_index: the selected index of the output, in [0, output_dim) .
    num_pts: num of points sampled for the curve.
    num_extrapolate_bins (N_e): num of bins extrapolating from the given grids. The
        curve will be calculate in the range of
        [grid_min - h * N_e, grid_max + h * N_e].
    """  # noqa: D404
    ng = self.rbf.num_grids
    h = self.rbf.denominator
    assert input_index < self.input_dim
    assert output_index < self.output_dim
    w = self.spline_linear.linear.weight[
      output_index, input_index * ng : (input_index + 1) * ng
    ]  # num_grids,
    x = jnp.linspace(
      self.rbf.grid_min - num_extrapolate_bins * h,
      self.rbf.grid_max + num_extrapolate_bins * h,
      num_pts,
    )  # num_pts, num_grids
    y = (w * self.rbf(x)).sum(-1)
    return x, y
