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

from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxtyping import PyTree
from typing import Any, Literal

from rich.console import Console


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
  method: Literal[None, "sigmoid", "softmax", "threshold"] = "sigmoid",
  threshold: float = 0.5,
) -> jax.Array:
  """Classify the array."""
  match method:
    case "sigmoid":
      return jax.nn.sigmoid(x) > threshold
    case "softmax":
      x = jax.nn.softmax(x)
      return jnp.argmax(x, axis=-1, keepdims=True)
    case "threshold":
      return x > threshold
    case _:
      raise ValueError(f"Unknown method {method}")


def pformat(xs: PyTree, short_arrays: bool = False) -> str:
  """Pretty format."""
  with (console := Console()).capture() as capture:
    nxs = eqx.tree_pformat(xs, short_arrays=short_arrays)
    console.print(nxs, soft_wrap=True, justify="left", no_wrap=True, width=40)
  return capture.get()


def filter_concat[T](
  xs: Sequence[T],
  filter_spec: Callable[[Any], bool] = eqx.is_array,
  select_idx: int = -1,
) -> T:
  """Filter concat."""
  t1, t2 = eqx.partition(xs, filter_spec=filter_spec)
  t1 = jax.tree.map(lambda *x: jnp.r_[*x], *t1)
  return eqx.combine(t1, t2[select_idx])


def all_array(x: PyTree) -> list[jax.Array]:
  """All array."""
  return jax.tree.leaves(eqx.filter(x, eqx.is_array))
