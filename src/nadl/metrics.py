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
filename : metrics.py
project  : nadl
license  : GPL-3.0+

* Metric class and functions
* Some simple metrics that not in optax loss.  For complex metrics, use sklearn.
"""

import operator
from abc import abstractmethod
from collections.abc import Callable

import jax
import jax.numpy as jnp
from equinox import (
  AbstractVar,
  Module,
  combine,
  field,
  is_array,
  partition,
  tree_at,
  tree_equal,
  tree_pformat,
)

from jaxtyping import Array, Int, Num
from typing import Literal, Self

from .utils import filter_concat

type N = Num[Array, "#a"]  # noqa: F722


def convert(x: float | Num | None) -> N:
  """Convert to float."""
  if x is None:
    return jnp.asarray(jnp.nan).reshape(-1)
  if isinstance(x, jax.Array):
    return x.reshape(-1)
  return jnp.asarray(x).reshape(-1)


class AbstractMetric(Module):
  """Abstract Metric."""

  name: AbstractVar[str | None]

  @abstractmethod
  def __or__(self, value: Self) -> Self:
    """Or."""
    raise NotImplementedError

  @abstractmethod
  def __add__(self, value: Self) -> Self:
    """Add."""
    raise NotImplementedError

  @abstractmethod
  def best(self) -> Self:
    """Best value."""
    raise NotImplementedError

  def __repr__(self) -> str:
    """Representation."""
    return tree_pformat(self, short_arrays=False)

  @abstractmethod
  def __getitem__(self, idx: int | Int[Array, "#b"]) -> Self:  # noqa: F722
    """Get item."""
    raise NotImplementedError


class Metric(AbstractMetric):
  """Base Metric."""

  value: N = field(default=jnp.nan, converter=convert)
  order: Literal["max", "min"] = "max"
  name: str | None = None

  def __or__(self, value: Self) -> Self:
    """Or."""
    s1, _ = partition(self, lambda x: is_array(x) and (not jnp.isnan(x).any()))
    return combine(s1, value)

  def __add__(self, value: Self) -> Self:
    """Add."""
    if self.name != value.name:
      raise ValueError(f"Name not match: this {self.name=} != {value.name=}")
    if self.order != value.order:
      raise ValueError(f"Order not match: this {self.order=} != {value.order=}")
    return filter_concat([self, value])

  def _max(self) -> Int[Array, "1"]:
    """Best value."""
    return jnp.nanargmax(self.value)

  def _min(self) -> Int[Array, "1"]:
    """Worst value."""
    return jnp.nanargmin(self.value)

  def best(self) -> Self:
    """Best value."""
    return self[self.best_idx()]

  def best_idx(self) -> Int[Array, "1"]:
    """Best value."""
    return self._max() if self.order == "max" else self._min()

  def __repr__(self) -> str:
    """Representation."""
    return tree_pformat(self, short_arrays=False)

  def __getitem__(self, idx: int | Int[Array, "#b"]) -> Self:  # noqa: F722
    """Get item."""
    if self.value is None:
      raise ValueError("No value.")
    return tree_at(lambda x: x.value, self, self.value[idx])


class GroupMetric(AbstractMetric):
  """Group Metric."""

  metrics: list[Metric]
  name: str | None = None

  def __or__(self, value: Self) -> Self:
    """Or."""
    if tree_equal(*jax.tree.map(jnp.shape, [self.metrics, value.metrics])):
      s1, _ = partition(self, lambda x: is_array(x) and (not jnp.isnan(x).all()))
      return combine(s1, value)
    raise ValueError("Shape not match.")

  def __add__(self, value: Self) -> Self:
    """Add."""
    if self.name != value.name:
      raise ValueError(f"Name not match: this {self.name=} != {value.name=}")
    return filter_concat([self, value])

  def __getitem__(self, idx: int | Array) -> Self:
    """Get item."""
    return tree_at(
      lambda x: x.metrics, self, jax.tree.map(operator.itemgetter(idx), self.metrics)
    )

  def best(self, which: int | Callable[[list[Metric]], Metric]) -> Self:
    """Best value."""
    return self[self.best_idx(which)]

  def best_idx(self, which: int | Callable[[list[Metric]], Metric]) -> Int[Array, "1"]:
    """Best value."""
    match which:
      case Callable():
        return which(self.metrics).best_idx()
      case int():
        return self.metrics[which].best_idx()
      case _:
        raise TypeError("which should be int or list[m] -> m.")


def dice_coef(y_true: jax.Array, y_pred: jax.Array, eps: float = 1e-8) -> jax.Array:
  """Compute dice coefficient."""
  y_true = jnp.asarray(y_true)
  y_pred = jnp.asarray(y_pred)

  intersection = jnp.sum(y_true * y_pred)
  union = jnp.sum(y_true) + jnp.sum(y_pred)

  return (2.0 * intersection) / (union + eps)


def iou_coef(y_true: jax.Array, y_pred: jax.Array, eps: float = 1e-8) -> jax.Array:
  """Compute intersection over union."""
  y_true = jnp.asarray(y_true)
  y_pred = jnp.asarray(y_pred)

  intersection = jnp.sum(y_true * y_pred)
  union = jnp.sum(y_true) + jnp.sum(y_pred) - intersection

  return intersection / (union + eps)
