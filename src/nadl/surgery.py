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
date     : Jun 17, 2024
email    : Nasy <nasyxx+python@gmail.com>
filename : surgery.py
project  : nadl
license  : GPL-3.0+

Surgery for NADL.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxtyping import Array, Float, PRNGKeyArray
from typing import Any, Callable, TypeGuard

type F = Float[Array, " *"]
type K = PRNGKeyArray


def init_fn(fn: Callable[[K, tuple[int, ...]], F]) -> Callable[[F, K], F]:
  """Initialize function."""

  def _init_fn(weight: F, key: K) -> F:
    return fn(key, jnp.shape(weight))

  return _init_fn


def kaiming_init(weight: F, key: K) -> F:
  """Kaiming initialization."""
  return init_fn(jax.nn.initializers.he_normal())(weight, key)


def is_linear(x: Any) -> TypeGuard[eqx.nn.Linear]:  # noqa: ANN401
  """Check if a module is a linear layer."""
  return isinstance(x, eqx.nn.Linear)


def is_conv(x: Any) -> TypeGuard[eqx.nn.Conv]:  # noqa: ANN401
  """Check if a module is a convolution layer."""
  return isinstance(x, eqx.nn.Conv)


def is_conv1d(x: Any) -> TypeGuard[eqx.nn.Conv1d]:  # noqa: ANN401
  """Check if a module is a 1D convolution layer."""
  return isinstance(x, eqx.nn.Conv1d)


def is_conv2d(x: Any) -> TypeGuard[eqx.nn.Conv2d]:  # noqa: ANN401
  """Check if a module is a 2D convolution layer."""
  return isinstance(x, eqx.nn.Conv2d)


def get_weight(x: Any) -> Array:  # noqa: ANN401
  """Get weight of a module."""
  return x.weight


def get_bias(x: Any) -> Array:  # noqa: ANN401
  """Get bias of a module."""
  return x.bias


def init_surgery[T, M](
  model: T,
  key: K,
  is_leaf: Callable[[M], bool] = is_linear,
  init: Callable[[F, K], F] = kaiming_init,
  get_weight: Callable = get_weight,
) -> T:
  """Initialize model."""

  def _get_weights(m: Any) -> list[F]:  # noqa: ANN401
    return list(
      map(
        get_weight,
        filter(is_leaf, jax.tree.leaves(m, is_leaf=is_leaf)),
      )
    )

  weights = _get_weights(model)
  new_weights = list(map(init, weights, jax.random.split(key, len(weights))))
  return eqx.tree_at(_get_weights, model, new_weights)
