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
from functools import partial

import jax
import jax.numpy as jnp

from typing import Literal, TypeAlias

SCALER: TypeAlias = Callable[[jax.Array], jax.Array]


def identity_scaler(_arr: jax.Array, _axis: int = 0) -> SCALER:
  """Get identity scaler."""

  def scaler(x: jax.Array) -> jax.Array:
    """Scaler."""
    return x

  return scaler


def min_max_scaler(arr: jax.Array, axis: int = 0) -> SCALER:
  """Get min max scaler."""
  min_, max_ = arr.min(axis=axis, keepdims=True), arr.max(axis=axis, keepdims=True)

  def scaler(x: jax.Array) -> jax.Array:
    """Scaler."""
    return (x - min_) / (max_ - min_)

  return scaler


def standard_scaler(arr: jax.Array, axis: int = 0) -> SCALER:
  """Get standard scaler."""
  mean, std = arr.mean(axis=axis, keepdims=True), arr.std(axis=axis, keepdims=True)

  def scaler(x: jax.Array) -> jax.Array:
    """Scaler."""
    return (x - mean) / std

  return scaler


def normalizer(
  arr: jax.Array, axis: int = 0, norm: Literal["l1", "l2", "max"] = "l2"
) -> SCALER:
  """Get normalizer."""
  match norm:
    case "l2":
      norm_value = jnp.sqrt(jnp.sum(jnp.square(arr), axis=axis, keepdims=True))
    case "l1":
      norm_value = jnp.sum(jnp.abs(arr), axis=axis, keepdims=True)
    case "max":
      norm_value = jnp.max(jnp.abs(arr), axis=axis, keepdims=True)
    case _:
      raise ValueError("norm should be 'l1', 'l2', or 'max'")

  def scaler(x: jax.Array) -> jax.Array:
    """Scaler."""
    max_val = jnp.maximum(norm_value, jnp.finfo(x.dtype).tiny)  # Avoid division by zero
    return x / max_val

  return scaler


def scaler_fn(
  method: Literal["id", "minmax", "std", "l2_norm", "l1_norm", "max_norm"] = "minmax"
) -> Callable[[jax.Array, int], SCALER]:
  """Get scaler function."""
  match method:
    case "id":
      return identity_scaler
    case "minmax":
      return min_max_scaler
    case "std":
      return standard_scaler
    case "l2_norm":
      return partial(normalizer, norm="l2")
    case "l1_norm":
      return partial(normalizer, norm="l1")
    case "max_norm":
      return partial(normalizer, norm="max")
    case _:
      raise ValueError(f"Unknown scaler method {method}")
