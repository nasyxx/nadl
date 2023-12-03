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
date     : Nov 29, 2023
email    : Nasy <nasyxx+python@gmail.com>
filename : utils.py
project  : nadl
license  : GPL-3.0+

Utils
"""
from typing import Literal
import jax
import jax.numpy as jnp


def rle_array(x: jax.Array, shift: int = 1) -> jax.Array:
  """Run length encoding array."""
  x = x.flatten()
  x = jnp.pad(x, (1, 1), mode="constant")
  x = jnp.argwhere(x[1:] != x[:-1]).flatten() + shift
  return x.at[1::2].add(-x[::2])


def rle(x: jax.Array, shift: int = 1) -> str:
  """Run length encoding."""
  return " ".join(map(str, rle_array(x, shift)))


def classit(
  x: jax.Array,
  method: Literal[None, "sigmoid", "softmax"] = "sigmoid",
  keepdims: bool = True,
) -> jax.Array:
  """Classify the array."""
  match method:
    case "sigmoid":
      return jnp.where(jax.nn.sigmoid(x) > 0.5, 1, 0)  # noqa: PLR2004
    case "softmax":
      x = jax.nn.softmax(x)
      return jnp.argmax(x, axis=-1, keepdims=keepdims)
    case None:
      return x
    case _:
      raise ValueError(f"Unknown method {method}")
