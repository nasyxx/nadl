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
date     : Dec  6, 2023
email    : Nasy <nasyxx+python@gmail.com>
filename : preprocessing.py
project  : nadl
license  : GPL-3.0+

Preprocessing
"""
from collections.abc import Callable

import jax
import jax.numpy as jnp

from typing import Literal, TypeAlias


SCALER: TypeAlias = Callable[[jax.Array], jax.Array]


def min_max_scaler(arr: jax.Array) -> SCALER:
  """Get min max scaler."""
  min_, max_ = arr.min(), arr.max()

  def scaler(x: jax.Array) -> jax.Array:
    """Scaler."""
    return (x - min_) / (max_ - min_)

  return scaler


def standard_scaler(arr: jax.Array) -> SCALER:
  """Get standard scaler."""
  mean, std = arr.mean(), arr.std()

  def scaler(x: jax.Array) -> jax.Array:
    """Scaler."""
    return (x - mean) / std

  return scaler


def normalizer(arr: jax.Array, norm: Literal["l1", "l2", "max"] = "l2") -> SCALER:
  """Get normalizer."""
  match norm:
    case "l2":
      norm_value = jnp.sqrt(jnp.sum(jnp.square(arr)))
    case "l1":
      norm_value = jnp.sum(jnp.abs(arr))
    case "max":
      norm_value = jnp.max(jnp.abs(arr))
    case _:
      raise ValueError("norm should be 'l1', 'l2', or 'max'")

  def scaler(x: jax.Array) -> jax.Array:
    """Scaler."""
    max_val = jnp.maximum(norm_value, jnp.finfo(x.dtype).tiny)  # Avoid division by zero
    return x / max_val

  return scaler
