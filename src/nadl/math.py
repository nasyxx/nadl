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
date     : Sep 19, 2024
email    : Nasy <nasyxx+python@gmail.com>
filename : math.py
project  : nadl
license  : GPL-3.0+

Math operators.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array

EPS = 1e-15
HALF_EPS = 1e-7


def safe_div(x: Array, y: Array, eps: float = EPS) -> Array:
  """Safe division."""
  return jnp.divide(x, jnp.maximum(y, eps))


def safe_sqrt(x: Array, eps: float = EPS) -> Array:
  """Safe square root."""
  return jnp.sqrt(jnp.maximum(x, eps))


def inner(x: Array, y: Array, axis: int = -1, keepdims: bool = False) -> Array:
  """Inner product."""
  if x.ndim == 0:
    x = x[None]
  return jnp.sum(x * y, axis=axis, keepdims=keepdims)


def sq_norm(x: Array, axis: int = -1, keepdims: bool = False) -> Array:
  """Square norm."""
  if x.ndim == 0:
    x = x[None]
  return inner(x, x, axis=axis, keepdims=keepdims)


def norm(x: Array, axis: int = -1, keepdims: bool = False) -> Array:
  """Norm."""
  return safe_sqrt(sq_norm(x, axis=axis, keepdims=keepdims))


def sgn(x: Array) -> Array:
  """Symbol function."""
  return jax.lax.cond(x > 0, lambda: 1, lambda: -1, x)
