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

# ruff: noqa: F722
from functools import partial
from warnings import warn

import jax
import jax.numpy as jnp
from equinox import (
  AbstractVar,
  Module,
  combine,
  filter_pure_callback,
  partition,
  tree_equal,
  tree_pformat,
)

from jaxtyping import Array, ArrayLike, Int, Num
from typing import Literal, Self

import numpy as np
import sklearn.metrics as m

from .utils import batch_array_p, filter_concat, filter_tree

type N = Num[Array, "..."]


def convert(x: ArrayLike) -> N:
  """Convert to float."""
  x = jnp.asarray(x)
  return x.reshape(-1, *x.shape[1:])


def roc_auc_score(labels: Num[Array, " A"], preds: Num[Array, " A"]) -> N:
  """Compute ROC."""
  return jax.pure_callback(
    m.roc_auc_score, jax.ShapeDtypeStruct((), jnp.float32), labels, preds
  )


def average_precision_score(
  labels: Num[Array, " A"],
  preds: Num[Array, " A"],
  average: Literal["micro", "macro"] = "macro",
) -> N:
  """Compute PR."""
  return jax.pure_callback(
    partial(m.average_precision_score, average=average),
    jax.ShapeDtypeStruct((), jnp.float32),
    labels,
    preds,
  )


def pr_auc_score(labels: Num[Array, " A"], preds: Num[Array, " A"]) -> N:
  """Compute PR."""

  def _callback(lbl: Num[Array, " A"], prd: Num[Array, " A"]) -> N:
    precision, recall, _ = m.precision_recall_curve(lbl, prd)
    return jnp.asarray(m.auc(recall, precision))

  return filter_pure_callback(
    _callback,
    labels,
    preds,
    result_shape_dtypes=jax.ShapeDtypeStruct((), jnp.float32),
  )


class Metric[**P, T](Module):
  """Base Metric.

  This metric is for batch data not for a single one.
  Thus, please consider the shape of the input data as [b, ...].

  The default getitem will only take the array with ndim > 1.
  """

  name: AbstractVar[str | None]

  def __post_init__(self) -> None:
    """Post init."""
    for k, v in self.__dict__.items():
      if isinstance(v, Array | np.ndarray):
        self.__dict__[k] = convert(v)

  def __check_init__(self) -> None:  # noqa: PLW3201
    """Check init."""
    arrs = filter_tree(self, batch_array_p)
    if arrs:
      if not tree_equal(*jax.tree.map(lambda x: jnp.shape(x)[0], arrs)):
        raise ValueError("All batched array should have the same batch size.")
    else:
      warn("No batched array found in the metric.", stacklevel=1)

  @classmethod
  def merge(cls, *metrics: Self) -> Self:
    """Merge all metrics."""
    _, s2 = partition(metrics, batch_array_p)
    if not tree_equal(*s2):
      raise ValueError("All metrics should have the same non-array values.")
    return filter_concat(metrics, batch_array_p)

  def __add__(self, value: Self) -> Self:
    """Add."""
    return self.merge(self, value)

  def __or__(self, value: Self) -> Self:
    """Or."""
    return combine(self, value)

  @classmethod
  def empty(cls, *_args: P.args, **_kwds: P.kwargs) -> Self:
    """Empty."""
    return cls(**dict.fromkeys(cls.__dataclass_fields__))

  def compute(self) -> T:
    """Compute."""
    raise NotImplementedError

  def __getitem__(self, idx: int | slice | Int[ArrayLike, "..."]) -> Self:
    """Get item.

    The default getitem will only take the array with ndim > 1.
    """
    return jax.tree.map(lambda x: x[idx] if batch_array_p(x) else x, self)

  def show(self) -> str:
    """Show."""
    return tree_pformat(self, short_arrays=False)


class Accuracy(Metric):
  """Accuracy."""

  labels: Int[Array, "..."]
  preds: Int[Array, "..."]
  name: str = "accuracy"

  def compute(self) -> Array:
    """Compute."""
    return jnp.nanmean(self.labels == self.preds, axis=-1)


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
