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

from typing import Any, Callable, TypeGuard
import jax
import jax.numpy as jnp
import equinox as eqx
from .typings import F, K


def init_fn(fn: Callable[[K, tuple[int, ...]]]) -> Callable[[F, K], F]:
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


def init_surgery[T, M](
  model: T,
  key: K,
  is_leaf: Callable[[M], bool] = is_linear,
  init: Callable[[F, K], F] = kaiming_init,
) -> T:
  """Initialize model."""

  def _get_weights(m: Any) -> list[F]:  # noqa: ANN401
    return list(
      map(
        lambda x: x.weight,
        filter(is_leaf, jax.tree_util.tree_leaves(m, is_leaf=is_leaf)),
      )
    )

  weights = _get_weights(model)
  new_weights = list(map(init, weights, jax.random.split(key, len(weights))))
  return eqx.tree_at(_get_weights, model, new_weights)
